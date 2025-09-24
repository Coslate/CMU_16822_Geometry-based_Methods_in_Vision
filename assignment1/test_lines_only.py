import argparse, sys
from pathlib import Path
import numpy as np
import cv2

def keep_red_lines(img, dilate_px=0):
    # BGR mask (our lines were drawn as pure red: (0,0,255) in BGR)
    b, g, r = cv2.split(img)
    mask_bgr = (r > 200) & (g < 70) & (b < 70)

    # HSV mask (more robust to JPG compression / antialiasing)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # red wraps around 0/180: catch both sides
    mask_hsv1 = (h <= 10) & (s > 80) & (v > 80)
    mask_hsv2 = (h >= 170) & (s > 80) & (v > 80)

    mask = (mask_bgr | mask_hsv1 | mask_hsv2).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)

    out = np.full_like(img, 255)      # white background
    out[mask > 0] = img[mask > 0]     # keep only (red) test lines
    return out

def process_dir(outdir: Path, in_name="01_input_annotated.png",
                rect_name="03_rectified_with_lines.png",
                out_in="01b_test_lines_input.png",
                out_rect="03b_test_lines_rectified.png",
                dilate=0):
    ok = True
    inp = outdir / in_name
    rec = outdir / rect_name
    if not inp.exists():
        print(f"Error: {inp} not found", file=sys.stderr); ok = False
    if not rec.exists():
        print(f"Error: {rec} not found", file=sys.stderr); ok = False
    if not ok:
        return False

    img_in  = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    img_rec = cv2.imread(str(rec), cv2.IMREAD_COLOR)
    if img_in is None or img_rec is None:
        print("Error: Failed to read one of the images.", file=sys.stderr)
        return False

    only_in  = keep_red_lines(img_in,  dilate_px=dilate)
    only_rec = keep_red_lines(img_rec, dilate_px=dilate)

    cv2.imwrite(str(outdir / out_in), only_in)
    cv2.imwrite(str(outdir / out_rect), only_rec)
    print(f"Wrote {outdir/out_in} and {outdir/out_rect}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", help="output folder produced by affine_rectify.py (e.g., out_q1_book1)")
    ap.add_argument("--dilate", type=int, default=0, help="optional line thickening in pixels")
    ap.add_argument("--in-name", default="01_input_annotated.png")
    ap.add_argument("--rect-name", default="03_rectified_with_lines.png")
    ap.add_argument("--out-in", default="01b_test_lines_input.png")
    ap.add_argument("--out-rect", default="03b_test_lines_rectified.png")
    args = ap.parse_args()
    d = Path(args.outdir)
    if not d.exists():
        print(f"Error: {d} does not exist.", file=sys.stderr); sys.exit(1)
    ok = process_dir(d, args.in_name, args.rect_name, args.out_in, args.out_rect, args.dilate)
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()