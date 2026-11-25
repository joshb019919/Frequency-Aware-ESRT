"""Benchmark an ESRT variant on fixed random inputs (no data or checkpoints needed).

Role in ESRT: speed/memory/size comparison tool. Reports parameter count,
inference latency and one training step for the baseline model/esrt.py or an
altered variant (e.g. the frequency-aware changes), so variants can be compared.

  python bench.py --scale 4
  python bench.py --scale 4 --model_file /path/to/altered_esrt.py --json out.json

--model_file must define class ESRT(upscale=...). It is loaded as a standalone
module placed inside the repo's `model` package namespace, so relative imports
such as `from model import common` still work.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def load_esrt(path):
    """Return the ESRT class from `path`, or from model/esrt.py when `path` is None."""
    if path is None:
        from model import esrt
        return esrt.ESRT
    spec = importlib.util.spec_from_file_location("esrt_variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ESRT


def sync(dev):
    """Block until queued CUDA work finishes (no-op on CPU) so timings are accurate."""
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, warmup, iters):
    """Time `fn()` and return latency stats in milliseconds.

    Runs `warmup` untimed calls (cuDNN autotune, allocator warm-up), then
    `iters` timed calls, each followed by a device sync.

    Returns:
        dict with mean_ms, median_ms, min_ms, max_ms.
    """
    for _ in range(warmup):
        fn()
    sync(dev)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "median_ms": times[len(times) // 2],
            "min_ms": times[0], "max_ms": times[-1]}


def main():
    """Measure parameter count, inference latency and one train step; print/save JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--model_file", default=None)
    p.add_argument("--lr_size", type=int, nargs=2, default=[64, 64], help="H W of LR inference input")
    p.add_argument("--patch_size", type=int, default=96, help="HR patch size for train step")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    from model import esrt as _esrt
    import utils
    _esrt.add_model_args(p)
    utils.add_fast_arg(p)
    p.add_argument('--no_amp', action='store_true', help='plain fp32 (no bf16 autocast / channels_last / cudnn.benchmark)')
    p.add_argument("--json", default=None, help="write results to this file")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    try:
        model = load_esrt(a.model_file)(upscale=a.scale, **_esrt.model_kwargs(a)).to(dev)
    except ValueError as e:
        p.error(str(e))
    amp = dev.type == 'cuda' and not a.no_amp
    utils.enable_tf32(amp)
    torch.backends.cudnn.benchmark = amp
    if amp:
        model = model.to(memory_format=torch.channels_last)
    cl = torch.channels_last if amp else torch.contiguous_format
    res = {"amp_channels_last": amp, "fast": a.fast, "model_file": a.model_file or "model/esrt.py", "scale": a.scale, "freq": a.freq, "conv": a.conv, "global_res": a.global_res, "device": str(dev),
           "params": sum(x.numel() for x in model.parameters())}

    # Inference latency (batch 1, fixed LR size)
    model.eval()
    x = torch.rand(1, 3, *a.lr_size, device=dev).contiguous(memory_format=cl)
    infer_model = model
    if a.fast and dev.type == 'cuda':
        infer_model = utils.compile_model(model, x, autocast=True, mode=a.compile_mode)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(a.fast and dev.type == 'cuda')):
        res["inference"] = timeit(lambda: infer_model(x), dev, a.warmup, a.iters)
    res["inference"]["lr_size"] = a.lr_size
    if dev.type == "cuda":
        res["inference"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    # One training step (forward + L1 + backward + Adam)
    model.train()
    lr_p = a.patch_size // a.scale  # LR patch size matching the HR patch
    xb = torch.rand(a.batch_size, 3, lr_p, lr_p, device=dev).contiguous(memory_format=cl)
    yb = torch.rand(a.batch_size, 3, lr_p * a.scale, lr_p * a.scale, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, fused=dev.type == 'cuda')
    train_model = model
    if a.fast and dev.type == 'cuda':
        train_model = utils.compile_model(model, xb, train=True, autocast=amp, mode=a.compile_mode)
    crit = nn.L1Loss()

    def step():
        """A single optimisation step on the fixed random batch."""
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = train_model(xb)
        crit(out.float(), yb).backward()
        opt.step()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res["train_step"] = timeit(step, dev, a.warmup, a.iters)
    res["train_step"].update(batch_size=a.batch_size, patch_size=a.patch_size)
    if dev.type == "cuda":
        res["train_step"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
