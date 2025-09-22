from PIL import Image, ImageOps
from pathlib import Path
import argparse, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")

# --- use TA helpers ---
from utils import normalize, MyWarp, cosine, annotate

EPS = 1e-12

def to_h(pt_xy):
    return np.array([pt_xy[0], pt_xy[1], 1.0], dtype=float)

def line_from_pts(p, q):
    # p,q homogeneous points -> line l = p x q (homogeneous), then normalize via TA helper
    l = np.cross(p, q)
    return normalize(l)

def intersect_lines(l1, l2):
    v = np.cross(l1, l2)
    # normalize homogeneous point
    if abs(v[2]) < EPS:
        # still fine; normalize to unit length to avoid inf/nan scaling
        return normalize(v)
    return v / v[2]  # make z=1 for stability when possible

def group_pairs(points_xy):
    """
    points_xy: (16,2). Lines are (0,1),(2,3),...,(14,15).
    Parallel groups used to compute H: [(0,1),(2,3)]
    Hold-out groups for evaluation: [(4,5),(6,7)]
    """
    L = []
    for i in range(0, 16, 2):
        p = to_h(points_xy[i])
        q = to_h(points_xy[i+1])
        L.append(line_from_pts(p, q))
    compute_groups = [(0,1), (2,3)]
    holdout_groups = [(4,5), (6,7)]
    return L, compute_groups, holdout_groups

def build_affine_rectifier(vanishing_line):
    # map image vanishing line to [0,0,1]^T using a 1-DOF projective H
    l = normalize(vanishing_line)
    if abs(l[2]) < EPS:
        # rescale to avoid division by zero
        l = l / (np.linalg.norm(l[:2]) + EPS)
        l = np.array([l[0], l[1], 1.0], dtype=float)
    H = np.eye(3, dtype=float)
    H[2,0] = l[0] / l[2]
    H[2,1] = l[1] / l[2]
    return H

def compute_canvas_transform(H, w, h):
    """Replicates the translation step inside MyWarp so we can also warp points for visualization."""
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64).reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, H)
    xmin, ymin = (pts.min(axis=0).ravel() - 0.5).astype(int)
    xmax, ymax = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=float)
    return Ht @ H

def draw_pair(img, pts_xy, pair, color, thickness=2):
    p = tuple(np.int32(pts_xy[2*pair[0]]))
    q = tuple(np.int32(pts_xy[2*pair[0]+1]))
    cv2.line(img, p, q, color, thickness, cv2.LINE_AA)
    p2 = tuple(np.int32(pts_xy[2*pair[1]]))
    q2 = tuple(np.int32(pts_xy[2*pair[1]+1]))
    cv2.line(img, p2, q2, color, thickness, cv2.LINE_AA)

def pair_abs_cos(pts_xy, pair):
    p1 = pts_xy[2*pair[0]]
    q1 = pts_xy[2*pair[0] + 1]
    p2 = pts_xy[2*pair[1]]
    q2 = pts_xy[2*pair[1] + 1]
    u = q1 - p1
    v = q2 - p2
    return abs(float(cosine(u, v)))  # TA helper

def _load_points_from_ann(path, key):
    arr = np.load(path, allow_pickle=True)
    # Typical .npy dict saved with np.save -> 0-d object array that needs .item()
    try:
        d = arr.item()
        return np.array(d[key], dtype=float)
    except Exception:
        # Also allow plain (16,2) arrays saved directly
        a = np.array(arr, dtype=float)
        if a.shape == (16, 2):
            return a
        raise ValueError(f"Annotation file {path} is not a dict or (16,2) array.")    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to image")
    ap.add_argument("--ann", default="annotation/q1_annotation.npy")
    ap.add_argument("--key", help="annotation key in the .npy dict (e.g., 'book1')")
    ap.add_argument("--interactive", action="store_true",
                    help="click 16 points (8 lines, consecutive points form lines; (0,1)&(2,3) parallel, (4,5)&(6,7) parallel, etc.)")
    ap.add_argument("--outdir", default="out_q1", help="output directory")
    ap.add_argument("--save-ann", default="data/annotation/user_q1_annotation.npy", help="Where to save/append interactive clicks as a dict {key: (16,2)}.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load image WITH EXIF rotation applied
    pil = ImageOps.exif_transpose(Image.open(args.img)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]    
    if img is None:
        raise FileNotFoundError(args.img)
    h, w = img.shape[:2]

    '''
    # get 16 points (x,y)
    if args.interactive:
        print("\nClick 16 points in order:")
        print("Lines are (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15).")
        print("Consecutive PAIRS of lines are parallel: (0,1)&(2,3), (4,5)&(6,7), (8,9)&(10,11), (12,13)&(14,15).")
        clicks = annotate(args.img)  # TA helper returns [x,y,1] entries
        pts = np.array(clicks, dtype=float)[:,:2]
        if pts.shape != (16,2):
            raise ValueError(f"Need exactly 16 clicks; got {pts.shape[0]}")
    else:
        # load from npy
        with open(args.ann, "rb") as f:
            ann = np.load(f, allow_pickle=True).item()
        if args.key is None:
            raise ValueError("Provide --key when using --ann")
        pts = np.array(ann[args.key], dtype=float)  # (16,2)
    '''

    # --- get 16 points (x,y) ---
    if args.interactive:
        print("\nClick 16 points in order:")
        print("Lines are (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15).")
        print("Consecutive PAIRS of lines are parallel: (0,1)&(2,3), (4,5)&(6,7), (8,9)&(10,11), (12,13)&(14,15).")

        clicks = annotate(args.img)  # returns [x,y,1.]
        pts = np.asarray(clicks, dtype=float)[:, :2]
        if pts.shape != (16, 2):
            raise ValueError(f"Need exactly 16 clicks; got {pts.shape[0]}")

        # Save the clicks into a dict .npy so it can be reused like the course annotation
        key_to_save = Path(args.img).stem  # e.g., "computer1"
        save_path = Path(args.save_ann)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            try:
                d = np.load(save_path, allow_pickle=True).item()
            except Exception:
                d = {}
        else:
            d = {}
        d[key_to_save] = pts.astype(np.float32)
        np.save(save_path, d)

        # (Optional) also drop raw copies next to outputs
        base = key_to_save
        np.savetxt(Path(args.outdir, f"{base}_pts_xy.txt"), pts, fmt="%.6f")
        np.save   (Path(args.outdir, f"{base}_pts_xy.npy"), pts.astype(np.float32))

        print(f"Saved points under key '{key_to_save}' in {save_path}")
        print(f"Re-run later with: --ann {save_path} --key {key_to_save}")
    else:
        if args.key is None:
            raise ValueError("Provide --key when using --ann")
        pts = _load_points_from_ann(args.ann, args.key)  # (16,2)

    # build lines & groups
    lines, compute_groups, holdout_groups = group_pairs(pts)

    # vanishing points & vanishing line
    vps = []
    for (i,j) in compute_groups:
        vps.append(intersect_lines(lines[i], lines[j]))
    v1, v2 = vps
    l_inf = np.cross(v1, v2)
    l_inf = normalize(l_inf)

    # rectifier
    H = build_affine_rectifier(l_inf)

    # warp with TA helper (image); replicate its translation to warp points for drawing
    rect = MyWarp(img, H)  # TA helper
    M = compute_canvas_transform(H, w, h)  # same transform used inside MyWarp

    # === DRAW & SAVE ===
    # 1) INPUT image — draw ONLY the lines used to compute H (compute groups)
    vis_in_compute = img.copy()
    draw_pair(vis_in_compute, pts, compute_groups[0], (0,255,0), 3)   # green
    draw_pair(vis_in_compute, pts, compute_groups[1], (255,0,0), 3)   # blue
    cv2.imwrite(os.path.join(args.outdir, "01_input_annotated.png"), vis_in_compute)

    # 2) INPUT image — draw ONLY the TEST (hold-out) lines
    vis_in_test = img.copy()
    draw_pair(vis_in_test, pts, holdout_groups[0], (0, 0, 255), 3)      # red
    draw_pair(vis_in_test, pts, holdout_groups[1], (0, 255, 0), 3)      # green
    cv2.imwrite(os.path.join(args.outdir, "01b_test_lines_input.png"), vis_in_test)

    # 3) RECTIFIED points (use OpenCV's 2D perspectiveTransform)
    pts_2d   = pts.reshape(-1, 1, 2).astype(np.float64)
    pts_rect = cv2.perspectiveTransform(pts_2d, M).reshape(-1, 2)
    cv2.imwrite(os.path.join(args.outdir, "02_rectified.png"), rect)

    # 4) RECTIFIED image — draw ONLY the lines used to compute H (compute groups)
    vis_rect_compute = rect.copy()
    draw_pair(vis_rect_compute, pts_rect, compute_groups[0], (0,255,0), 3)  # green
    draw_pair(vis_rect_compute, pts_rect, compute_groups[1], (255,0,0), 3)  # blue
    cv2.imwrite(os.path.join(args.outdir, "03_rectified_with_lines.png"), vis_rect_compute)

    # 5) RECTIFIED image — draw ONLY the TEST (hold-out) lines, two colors
    vis_rect_test = rect.copy()
    draw_pair(vis_rect_test, pts_rect, holdout_groups[0], (0, 0, 255), 3) # red
    draw_pair(vis_rect_test, pts_rect, holdout_groups[1], (0, 255, 0), 3) # green
    cv2.imwrite(os.path.join(args.outdir, "03b_test_lines_rectified.png"), vis_rect_test)

    # cosine eval with TA helper (absolute cosines)
    beforeA = pair_abs_cos(pts, holdout_groups[0])
    beforeB = pair_abs_cos(pts, holdout_groups[1])

    # For cosine after rectification, translation doesn't matter; directions come from transformed points
    afterA  = pair_abs_cos(pts_rect, holdout_groups[0])
    afterB  = pair_abs_cos(pts_rect, holdout_groups[1])

    # save
    with open(os.path.join(args.outdir, "04_cosines.txt"), "w") as f:
        f.write("Hold-out parallel pairs (|cos theta|)\n")
        f.write(f"Pair red lines: before = {beforeA:.6f}, after = {afterA:.6f}\n")
        f.write(f"Pair green lines: before = {beforeB:.6f}, after = {afterB:.6f}\n")

    print("Vanishing line (normalized):", l_inf)
    print(f"|cos| hold-outs: A {beforeA:.6f} → {afterA:.6f};  B {beforeB:.6f} → {afterB:.6f}")
    print("Saved outputs to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()