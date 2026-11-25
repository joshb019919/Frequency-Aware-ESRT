"""Build test.py-ready HR/LR folders for BSDS300 (test split) and Manga109.

test.py expects `<root>/HR/<name>.png` and `<root>/LR/x<s>/<name>x<s>.png`. BSDS300 ships
only full-size JPEGs and Manga109 only per-book page JPEGs, so this adds those two folders
inside each dataset directory. Existing files are never modified or deleted; outputs that
already exist are skipped (use --force to rewrite them).

  BSDS300 : the 100 images listed in iids_test.txt (images/test/<id>.jpg).
  Manga109: one page per book (the middle page, chosen deterministically) -> 109 images,
            named <Book>_<page>.png. This is NOT the official 109-image SR benchmark list.

LR = bicubic (PIL, antialiased) downscale of the HR image mod-cropped to a multiple of the
scale, matching what test.py/utils.modcrop do to the HR side.
"""
import argparse
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
        yield i, os.path.join(root, "images", "test", i + ".jpg")


def manga109_items(root):
    img_root = os.path.join(root, "images")
    for book in sorted(os.listdir(img_root)):
        pages = sorted(p for p in os.listdir(os.path.join(img_root, book)) if p.lower().endswith(".jpg"))
        if pages:
            page = pages[len(pages) // 2]
            yield "{}_{}".format(book, os.path.splitext(page)[0]), os.path.join(img_root, book, page)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="/data/data")
    p.add_argument("--sets", default="bsds300,manga109")
    p.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    a = p.parse_args()
    gens = {"bsds300": bsds300_items, "manga109": manga109_items}
    for name in [s for s in a.sets.split(",") if s]:
        if name not in gens:
            p.error("unknown set {!r}; choose from {}".format(name, ", ".join(gens)))
        root = os.path.join(a.data_root, name)
        written = count = 0
        for img_name, path in gens[name](root):
            written += write_pair(Image.open(path).convert("RGB"), img_name, root, SCALES, a.force)
            count += 1
        print("{}: {} images, {} files written under {}/{{HR,LR}}".format(name, count, written, root))


if __name__ == "__main__":
    main()
