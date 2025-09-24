import os
import numpy as np
import cv2
import argparse
from pathlib import Path
from affine_rectify import _load_points_from_ann, compute_canvas_transform
from utils import normalize, MyWarp, cosine, annotate

EPS = 1e-12

def to_h(p):
    p = np.asarray(p, float)
    if p.ndim == 1: return np.array([p[0], p[1], 1.0], float)
    return np.hstack([p, np.ones((p.shape[0], 1))])

def from_h(P):
    P = np.asarray(P, float)
    P = P / (P[..., 2:3] + EPS)
    return P[..., :2]

def line_from_pts(p, q):
    return np.cross(to_h(p), to_h(q))

def apply_H_to_points(H, pts_xy):
    P = to_h(pts_xy)
    Pp = (H @ P.T).T
    return from_h(Pp)

def draw_segments(img, segs, color=(0,255,0), thickness=2):
    vis = img.copy()
    for (p, q) in segs:
        p = tuple(np.round(p).astype(int))
        q = tuple(np.round(q).astype(int))
        cv2.line(vis, p, q, color, thickness, cv2.LINE_AA)
    return vis

def build_A_affine(line_pairs):
    rows = []
    for l, m in line_pairs:
        l = l / (np.linalg.norm(l[:2]) + EPS)
        m = m / (np.linalg.norm(m[:2]) + EPS)
        l1, l2 = l[0], l[1]
        m1, m2 = m[0], m[1]
        rows.append([l1*m1, (l1*m2 + l2*m1), l2*m2])  # a, b, c coefficients
    return np.asarray(rows, float)

def solve_Cstar_affine(line_pairs):
    # Solve A@[a,b,c]=0 (smallest singular vector)
    A = build_A_affine(line_pairs)
    _, _, Vt = np.linalg.svd(A)
    a, b, c = Vt[-1]
    s = np.array([a, b, c], float)
    if s[0] < 0: s = -s
    Cstar = np.array([[s[0], s[1], 0.0],
                      [s[1], s[2], 0.0],
                      [0.0 , 0.0 , 0.0]], float)
    return Cstar

def metric_H_from_Cstar_svd(Cstar, eps=1e-12):
    """
    Slide method (full 3x3):
      C*' = U diag(σ1, σ2, 0) U^T
      H   = diag(σ1^{-1/2}, σ2^{-1/2}, 1) U^T
    """
    # Symmetrize for numerical stability
    Cstar = 0.5 * (Cstar + Cstar.T)
    U, s, _ = np.linalg.svd(Cstar)          # s sorted desc: s[0] >= s[1] >= s[2] ~ 0
    s1 = max(float(s[0]), eps)
    s2 = max(float(s[1]), eps)
    Dinv_sqrt = np.diag([1.0/np.sqrt(s1), 1.0/np.sqrt(s2), 1.0])
    H = Dinv_sqrt @ U.T
    return H

def seg_cos(P1, P2, Q1, Q2):
    v1 = (P2 - P1).astype(float); v2 = (Q2 - Q1).astype(float)
    v1 /= (np.linalg.norm(v1) + EPS); v2 /= (np.linalg.norm(v2) + EPS)
    return float(np.dot(v1, v2))

def main():
    ap = argparse.ArgumentParser(description="Q2: Metric rectification given an affine-rectified image")
    ap.add_argument("--image", required=True, help="path to original input image")
    ap.add_argument("--ann", help="npy with 16x2 points: (0,1)&(2,3), (4,5)&(6,7) are perp pairs; (8..15) held-out")
    ap.add_argument("--key", help="annotation key if ann is a dict .npy")
    ap.add_argument("--interactive", action="store_true",
                    help="click 16 points (8 lines, consecutive points form lines; (0,1)&(2,3) perpendicular, (4,5)&(6,7) perpendicular, etc.)")
    ap.add_argument("--H_aff", required=True, help="path to out_q1_xxx/H_aff.npy")
    ap.add_argument("--save-ann", default="data/annotation/user_q2_annotation.npy", help="Where to save/append interactive clicks as a dict {key: (16,2)}.")
    ap.add_argument("--out", required=True, help="output directory (e.g., out_q2_floor1)")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    Himg, Wimg = img.shape[:2]
    H_aff = np.load(args.H_aff)

    # --- get 16 points (x,y) ---
    if args.interactive:
        print("\nClick 16 points in order:")
        print("Lines are (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15).")
        print("Consecutive PAIRS of lines are perpendicular: (0,1)&(2,3), (4,5)&(6,7), (8,9)&(10,11), (12,13)&(14,15).")

        clicks = annotate(args.image)  # returns [x,y,1.]
        pts = np.asarray(clicks, dtype=float)[:, :2]
        if pts.shape != (16, 2):
            raise ValueError(f"Need exactly 16 clicks; got {pts.shape[0]}")

        # Save the clicks into a dict .npy so it can be reused like the course annotation
        key_to_save = Path(args.image).stem  # e.g., "computer1"
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
        np.savetxt(Path(args.out, f"{base}_pts_xy.txt"), pts, fmt="%.6f")
        np.save   (Path(args.out, f"{base}_pts_xy.npy"), pts.astype(np.float32))

        print(f"Saved points under key '{key_to_save}' in {save_path}")
        print(f"Re-run later with: --ann {save_path} --key {key_to_save}")
    else:
        if args.key is None:
            raise ValueError("Provide --key when using --ann")
        pts = _load_points_from_ann(args.ann, args.key)  # (16,2)

    assert pts.shape == (16, 2), "Q2 annotation must be (16,2)"

    # Build segments & lines in original image
    segs = [(pts[i], pts[i+1]) for i in range(0, 16, 2)]
    lines = [line_from_pts(p, q) for (p, q) in segs]

    # Training perpendicular pairs: (0,1) and (2,3); Held-out: (4,5) and (6,7)
    train_pairs = [(lines[0], lines[1]), (lines[2], lines[3])]
    test_pairs  = [(segs[4], segs[5]), (segs[6], segs[7])]

    # 01_input_annotated.png (train only)
    vis_in = draw_segments(img, [segs[0], segs[1]], (0,255,0), 3)   # green
    vis_in = draw_segments(vis_in, [segs[2], segs[3]], (255,0,0), 3)  # blue
    cv2.imwrite(os.path.join(args.out, "01_input_annotated.png"), vis_in)

    # 01b_test_lines_input.png (held-out only)
    vis_in_test = draw_segments(img,  [segs[4], segs[5]], (0,0,255), 3)  # red
    vis_in_test = draw_segments(vis_in_test, [segs[6], segs[7]], (0,255,0), 3)  # green
    cv2.imwrite(os.path.join(args.out, "01b_test_lines_input.png"), vis_in_test)

    # Affine warp of points (for C* estimation on affine frame)
    pts_aff = apply_H_to_points(H_aff, pts)
    segs_aff = [(pts_aff[i], pts_aff[i+1]) for i in range(0, 16, 2)]
    lines_aff = [line_from_pts(p, q) for (p, q) in segs_aff]
    train_pairs_aff = [(lines_aff[0], lines_aff[1]), (lines_aff[2], lines_aff[3])]

    # Estimate C* and compute metric rectifying homography on affine frame
    Cstar_aff = solve_Cstar_affine(train_pairs_aff)
    H_met = metric_H_from_Cstar_svd(Cstar_aff)
    H_total = H_met @ H_aff

    # Final metric rectified image (padded) + rectified points with padding
    img_met = MyWarp(img, H_total)
    H_pad = compute_canvas_transform(H_total, Wimg, Himg)
    pts_met = apply_H_to_points(H_pad, pts)  # NOTE: use H_pad to align with padded canvas
    segs_met = [(pts_met[i], pts_met[i+1]) for i in range(0, 16, 2)]

    # 02_rectified.png (metric)
    cv2.imwrite(os.path.join(args.out, "02_rectified.png"), img_met)

    # 03_rectified_with_lines.png (train pairs on rectified)
    vis_rect_train = draw_segments(img_met, [segs_met[0], segs_met[1]], (0,255,0), 3)  # green
    vis_rect_train = draw_segments(vis_rect_train, [segs_met[2], segs_met[3]], (255,0,0), 3)  # blue
    cv2.imwrite(os.path.join(args.out, "03_rectified_with_lines.png"), vis_rect_train)

    # 03b_test_lines_rectified.png (held-out pairs on rectified)
    vis_rect_test = draw_segments(img_met, [segs_met[4], segs_met[5]], (0,0,255), 3)  # red
    vis_rect_test = draw_segments(vis_rect_test, [segs_met[6], segs_met[7]], (0,255,0), 3)  # green
    cv2.imwrite(os.path.join(args.out, "03b_test_lines_rectified.png"), vis_rect_test)

    # Evaluate |cos| on held-out pairs: before vs after
    beforeA = seg_cos(*test_pairs[0][0], *test_pairs[0][1])
    beforeB = seg_cos(*test_pairs[1][0], *test_pairs[1][1])
    afterA  = seg_cos(*segs_met[4], *segs_met[5])
    afterB  = seg_cos(*segs_met[6], *segs_met[7])

    with open(os.path.join(args.out, "04_cosines.txt"), "w") as f:
        f.write("Hold-out perpendicular pairs (|cos theta|)\n")
        f.write(f"Pair red lines: before = {beforeA:.6f}, after = {afterA:.6f}\n")
        f.write(f"Pair green lines: before = {beforeB:.6f}, after = {afterB:.6f}\n")

    # (Optional artifacts you may want to keep for debugging)
    np.save(os.path.join(args.out, "H_metric.npy"), H_met)
    np.save(os.path.join(args.out, "H_total.npy"),  H_total)
    print(f"|cos| hold-outs: A {beforeA:.6f} → {afterA:.6f};  B {beforeB:.6f} → {afterB:.6f}")

    print("Saved outputs to:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()