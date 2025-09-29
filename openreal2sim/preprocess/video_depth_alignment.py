#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess an existing sgd_cvd_hr.npz:
1) Resize all images/depths to the original first-frame RGB size.
2) (Optional) MoGe refinement per frame: run MoGe on resized images and align each to resized npz depths.
3) Compute a global (a,b) on frame-0 (using the refined depths if enabled) via robust IRLS to GT depth,
   then apply the same (a,b) to ALL frames.
4) Save resized JPGs to outputs/{key}/resized_images/{idx:06d}.jpg
   and overwrite sgd_cvd_hr.npz with:
      images (uint8, N,H,W,3), depths (float16, N,H,W),
      intrinsic K (3x3 from config), cam_c2w (identity, N,4,4).
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
import yaml

# NEW: MoGe imports
import torch
from moge.model.v2 import MoGeModel

# --------------------- paths / config ---------------------
base_dir = Path.cwd()

# --------------------- (kept) helpers ---------------------
def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber weights: w = 1 if |r| <= d else d/|r|."""
    abs_r = np.abs(residuals)
    w = np.ones_like(residuals, dtype=np.float64)
    mask = abs_r > delta
    w[mask] = (delta / (abs_r[mask] + 1e-12))
    return w

def robust_scale_shift_align(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    mask: np.ndarray,
    iters: int = 5,
    huber_delta: float = 0.02
) -> Tuple[float, float]:
    """
    Solve for a,b in:  a * pred_depth + b ≈ ref_depth, on the masked region.
    Uses Iteratively Reweighted Least Squares with Huber weights.
    """
    assert pred_depth.shape == ref_depth.shape == mask.shape
    valid = (mask > 0) & np.isfinite(pred_depth) & np.isfinite(ref_depth) & (pred_depth > 0) & (ref_depth > 0)
    if valid.sum() < 100:
        ratio = np.median(ref_depth[valid]) / (np.median(pred_depth[valid]) + 1e-12)
        return float(ratio), 0.0

    x = pred_depth[valid].astype(np.float64)
    y = ref_depth[valid].astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x, dtype=np.float64)

    for _ in range(iters):
        Aw = A * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        params, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        a, b = params[0], params[1]
        r = (A @ params) - y
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        sigma = 1.4826 * mad
        delta = huber_delta if sigma < 1e-12 else huber_delta * sigma / max(sigma, 1e-12)
        w = huber_weights(r - med, delta)

    return float(a), float(b)

def export_frame0_pointcloud_o3d(
    img_color_ref: np.ndarray,
    depth_ref_m: np.ndarray,
    depth_aligned0_m: np.ndarray,
    K: np.ndarray,
    out_dir: Path,
):
    """
    Export two PLYs for frame-0 using the ALIGNED predicted depth:
      1) depth_predict.ply  : aligned predicted depth colored by RGB
      2) depth_aligned.ply  : fused (GT colored + aligned predicted in blue)
    """
    import open3d as o3d
    assert img_color_ref.ndim == 3 and img_color_ref.shape[2] == 3
    H, W = depth_ref_m.shape
    assert depth_aligned0_m.shape == (H, W)
    assert K.shape == (3, 3)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    def depth_to_xyz_colors(depth_m: np.ndarray,
                            rgb_img_or_none: np.ndarray | None,
                            solid_rgb_if_none: tuple[int, int, int] | None):
        valid = np.isfinite(depth_m) & (depth_m > 0)
        z = depth_m[valid].astype(np.float32)
        x = (uu[valid] - cx) * z / fx
        y = (vv[valid] - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        if rgb_img_or_none is not None:
            cols = (rgb_img_or_none[valid].astype(np.float32) / 255.0)
        else:
            rgb = np.array(solid_rgb_if_none, dtype=np.float32) / 255.0
            cols = np.tile(rgb[None, :], (pts.shape[0], 1))
        return pts, cols

    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) Aligned predicted only
    pts_pred, col_pred = depth_to_xyz_colors(depth_aligned0_m, img_color_ref, None)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pts_pred.astype(np.float32))
    pcd_pred.colors = o3d.utility.Vector3dVector(col_pred.astype(np.float32))
    o3d.io.write_point_cloud(str(out_dir / "depth_predict.ply"), pcd_pred)

    # (2) Fused: GT + aligned predicted in blue
    pts_rs, col_rs = depth_to_xyz_colors(depth_ref_m, img_color_ref, None)
    pts_al, col_al = depth_to_xyz_colors(depth_aligned0_m, None, (0, 0, 255))
    pts_all = np.concatenate([pts_rs, pts_al], axis=0).astype(np.float32)
    col_all = np.concatenate([col_rs, col_al], axis=0).astype(np.float32)
    pcd_al = o3d.geometry.PointCloud()
    pcd_al.points = o3d.utility.Vector3dVector(pts_all)
    pcd_al.colors = o3d.utility.Vector3dVector(col_all)
    o3d.io.write_point_cloud(str(out_dir / "depth_aligned.ply"), pcd_al)

# --------------------- new small helpers ---------------------
def resize_images_to_size(images: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a stack of RGB images (N,h,w,3) to (N,target_h,target_w,3) using Lanczos."""
    out = []
    for im in images:
        out.append(cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4))
    return np.stack(out, axis=0).astype(np.uint8)

def resize_depths_to_size(depths: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a stack of depths (N,h,w) to (N,target_h,target_w) with bilinear interpolation."""
    out = []
    for d in depths:
        out.append(cv2.resize(d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0).astype(np.float32)

def save_jpegs(images_rgb: np.ndarray, out_dir: Path) -> None:
    """Save images as {idx:06d}.jpg in BGR order for OpenCV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    N = images_rgb.shape[0]
    for idx in range(N):
        path = out_dir / f"{idx:06d}.jpg"
        bgr = cv2.cvtColor(images_rgb[idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

def filter_depths(depths: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """Set the pixels outside [min_val, max_val] in the limits"""
    if min_val is not None:
        depths[depths < min_val] = min_val
    if max_val is not None:
        depths[depths > max_val] = max_val
    return depths.astype(np.float32)

def overwrite_npz(npz_path: Path, images: np.ndarray, depths: np.ndarray, K: np.ndarray) -> None:
    """Write images/depths/K and identity poses to sgd_cvd_hr.npz (overwrite)."""
    N = images.shape[0]
    cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=N, axis=0)
    np.savez(
        str(npz_path),
        images=images.astype(np.uint8),
        depths=depths.astype(np.float16),
        intrinsic=K.astype(np.float32),
        cam_c2w=cam_c2w.astype(np.float32),
    )

def run_moge_depth(img_rgb_u8: np.ndarray, device: torch.device, model: MoGeModel) -> np.ndarray:
    """
    Run MoGe-2 depth predictor on a single RGB image (uint8 HxWx3).
    Returns float32 depth (H,W) in arbitrary metric (to be aligned).
    """
    with torch.no_grad():
        inp = torch.tensor(img_rgb_u8 / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        out = model.infer(inp[0])  # {"depth": HxW}
        depth = out["depth"].detach().float().cpu().numpy()
    return depth

def maybe_refine_depths_with_mono(imgs_resized: np.ndarray, depths_resized: np.ndarray, refine: bool) -> np.ndarray:
    """
    If refine is True:
        For each frame t: run MoGe depth on imgs_resized[t], then robustly align it to depths_resized[t].
        Return the per-frame aligned MoGe depths as the refined stack.
    Else:
        Return depths_resized unchanged.
    """
    if not refine:
        return depths_resized

    print("[Note] Running per-frame MoGe depth refinement and alignment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    H, W = imgs_resized.shape[1:3]
    refined = []
    ones = np.ones((H, W), dtype=np.uint8)

    for t in range(imgs_resized.shape[0]):
        d_moge = run_moge_depth(imgs_resized[t], device, model)        # float32 (H,W)
        # Align MoGe to npz depth of the same frame
        a_t, b_t = robust_scale_shift_align(d_moge, depths_resized[t], ones, iters=5, huber_delta=0.02)
        refined.append((a_t * d_moge + b_t).astype(np.float32))

    return np.stack(refined, axis=0).astype(np.float32)

def process_key(key: str, K: np.ndarray, depth_scale: float, refine: bool) -> None:
    """
    For a given key:
      - Load original first-frame RGB/Depth to get target size & GT depth (meters)
      - Load npz (images, depths)
      - Resize images/depths to target size
      - (Optional) MoGe refinement: per-frame MoGe -> align to resized npz depth
      - Compute global (a,b) on frame-0 (using refined stack if enabled) via IRLS to GT
      - Apply (a,b) to ALL frames, clip, save JPGs, overwrite npz, export PLYs, export HTML
    """
    print(f"[Config] Processing key: {key}")

    # Paths
    recon_dir = base_dir / "outputs" / key / "geometry" / "reconstruction"
    npz_path = recon_dir / "sgd_cvd_hr.npz"
    out_jpg_dir = base_dir / "outputs" / key / "resized_images"
    img0_path = base_dir / "data" / f"{key}.png"
    depth0_path = base_dir / "data" / f"{key}_depth.png"

    # Load target RGB and GT depth (meters)
    img0_rgb = cv2.cvtColor(cv2.imread(str(img0_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tgt_H, tgt_W = img0_rgb.shape[:2]
    depth0_raw = cv2.imread(str(depth0_path), cv2.IMREAD_UNCHANGED)
    depth0_m = depth0_raw.astype(np.float32) * float(depth_scale)

    # depth0_m = filter_depths(depth0_m, max_val=1.6)  # clip the real depth for better alignment

    # Load npz
    data = np.load(str(npz_path))
    imgs_npz = data["images"]     # (N,h,w,3) uint8
    depths_npz = data["depths"]   # (N,h,w) float16/float32
    N = imgs_npz.shape[0]

    # Resize
    imgs_resized = resize_images_to_size(imgs_npz, tgt_H, tgt_W)         # uint8
    depths_resized = resize_depths_to_size(depths_npz, tgt_H, tgt_W)     # float32

    # Optional: MoGe refinement (per-frame align MoGe -> resized npz)
    depths_for_alignment = maybe_refine_depths_with_mono(imgs_resized, depths_resized, refine)

    # Global robust alignment on frame-0 (mask = all-ones), using depths_for_alignment
    pred0 = depths_for_alignment[0]
    gt0 = depth0_m
    ones_mask = np.ones_like(pred0, dtype=np.uint8)
    a, b = robust_scale_shift_align(pred0, gt0, ones_mask, iters=5, huber_delta=0.02)
    print(f"[{key}] global IRLS params from frame-0: a={a:.6f}, b={b:.6f} (refine={refine})")

    # Apply to all frames, clip and cast
    depths_aligned = a * depths_for_alignment + b
    depths_aligned = np.clip(depths_aligned, 1e-3, 1e2).astype(np.float16)

    # Save JPGs and overwrite NPZ
    save_jpegs(imgs_resized, out_jpg_dir)
    overwrite_npz(npz_path, imgs_resized, depths_aligned, K)

    # Export PLYs (depth_predict == aligned predicted depth)
    export_frame0_pointcloud_o3d(
        img0_rgb,
        depth0_m,
        (a * pred0 + b).astype(np.float32),  # already aligned predicted depth (frame-0)
        K,
        recon_dir,
    )

    # Optional HTML (keep your current import path)
    print(f"[Visualization] Saving multi-step PCD HTML to: {recon_dir / 'multistep_pcd.html'}")
    from utils.viz_dynamic_pcd import save_multistep_pcd_html
    save_multistep_pcd_html(key, max_points=5000)

    print(f"[{key}] resized JPGs -> {out_jpg_dir}")
    print(f"[{key}] overwritten NPZ -> {npz_path}")

# --------------------- main: read YAML and run ---------------------
if __name__ == "__main__":
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    refine = cfg.get("depth_refinement", False)

    for key in keys:
        fx = float(cfg["specs"][key]["intrinsics"]["fx"])
        fy = float(cfg["specs"][key]["intrinsics"]["fy"])
        cx = float(cfg["specs"][key]["intrinsics"]["cx"])
        cy = float(cfg["specs"][key]["intrinsics"]["cy"])
        depth_scale = float(cfg["specs"][key]["intrinsics"]["depth_scale"])

        K = np.array(
            [[fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]],
        dtype=np.float32)

        process_key(key, K, depth_scale, refine)
