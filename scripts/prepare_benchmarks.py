"""Build test.py-ready HR/LR folders for BSDS300 (test split) and Manga109.

test.py expects `<root>/HR/<name>.png` and `<root>/LR/x<s>/<name>x<s>.png`. BSDS300 ships
only full-size JPEGs and Manga109 only per-book page JPEGs, so this adds those two folders
inside each dataset directory. Existing files are never modified or deleted; outputs that
already exist are skipped (use --force to rewrite them).

  BSDS300 : the 100 images listed in iids_test.txt (images/test/<id>.jpg).
  Manga109: the 109-image SR benchmark (one image per book, named like the upstream ESRT outputs in
            result/Manga109/x2). Each is one 826x1170 half (right unless noted) of the book's 000.jpg
            spread, recovered by matching those outputs against the dataset (scripts/manga109_official.json;
            right half = columns 827..1652, 107 of 108 same-size books match at 32-41 dB). Exception:
            PrayerHaNemurenai's official image is 742x1072 and could not be located; the same rule is used.
            --manga109_pages middle selects the old stand-in (middle page of each book, full spread).

LR = bicubic (PIL, antialiased) downscale of the HR image mod-cropped to a multiple of the
scale, matching what test.py/utils.modcrop do to the HR side.
"""
import argparse
import json
import os
from PIL import Image

SCALES = (2, 3, 4)


def modcrop(img, s):
    w, h = img.size
    return img.crop((0, 0, w - w % s, h - h % s))


def write_pair(src_img, name, root, scales, force):
    """Write HR png and one LR png per scale; return number of files written."""
    n = 0
    hr_path = os.path.join(root, "HR", name + ".png")
    os.makedirs(os.path.dirname(hr_path), exist_ok=True)
    if force or not os.path.exists(hr_path):
        src_img.save(hr_path)
        n += 1
    for s in scales:
        lr_path = os.path.join(root, "LR", "x{}".format(s), "{}x{}.png".format(name, s))
        os.makedirs(os.path.dirname(lr_path), exist_ok=True)
        if force or not os.path.exists(lr_path):
            hr = modcrop(src_img, s)
            hr.resize((hr.width // s, hr.height // s), Image.BICUBIC).save(lr_path)
            n += 1
    return n


def bsds300_items(root):
    with open(os.path.join(root, "iids_test.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]
    for i in ids:
        yield i, os.path.join(root, "images", "test", i + ".jpg"), None


HALF_W, PAGE_H, RIGHT_X0 = 826, 1170, 827


def manga109_official_items(root):
    """Yield (name, path, crop box) for the 109 official benchmark images."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manga109_official.json")) as f:
        pages = json.load(f)
    for name in sorted(pages):
        e = pages[name]
        x0 = RIGHT_X0 if e["side"] == "R" else 0
        yield name, os.path.join(root, "images", e["dir"], e["page"]), (x0, 0, x0 + HALF_W, PAGE_H)


def manga109_middle_items(root):
    img_root = os.path.join(root, "images")
    for book in sorted(os.listdir(img_root)):
        pages = sorted(p for p in os.listdir(os.path.join(img_root, book)) if p.lower().endswith(".jpg"))
        if pages:
            page = pages[len(pages) // 2]
            yield "{}_{}".format(book, os.path.splitext(page)[0]), os.path.join(img_root, book, page), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="/data/data")
    p.add_argument("--sets", default="bsds300,manga109")
    p.add_argument("--manga109_pages", choices=["official", "middle"], default="official",
                   help="official = the 109-image SR benchmark pages (default); middle = old stand-in")
    p.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    a = p.parse_args()
    gens = {"bsds300": bsds300_items,
            "manga109": manga109_official_items if a.manga109_pages == "official" else manga109_middle_items}
    for name in [s for s in a.sets.split(",") if s]:
        if name not in gens:
            p.error("unknown set {!r}; choose from {}".format(name, ", ".join(gens)))
        root = os.path.join(a.data_root, name)
        written = count = 0
        for img_name, path, box in gens[name](root):
            img = Image.open(path).convert("RGB")
            written += write_pair(img.crop(box) if box else img, img_name, root, SCALES, a.force)
            count += 1
        print("{}: {} images, {} files written under {}/{{HR,LR}}".format(name, count, written, root))


if __name__ == "__main__":
    main()
