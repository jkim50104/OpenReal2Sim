#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full Mega-SAM depth & tracking pipeline.
- If --video_name is given: run once.
- Else: read keys from config/config.yaml and run all.
"""

import os
import argparse
import subprocess
from pathlib import Path
import yaml


def run(cmd_list, env):
    """Run a command and stream output; raise on non-zero exit."""
    print("Running:", " ".join(str(x) for x in cmd_list))
    subprocess.run(cmd_list, check=True, env=env)


def main(video_name: str, gpu: str):
    base_dir = Path.cwd()
    data_dir = base_dir / "outputs" / video_name
    frame_dir = data_dir / "images"
    depth_anything_dir = data_dir / "geometry" / "depthanything"
    uni_depth_dir = data_dir / "geometry" / "unidepth"
    recon_dir = data_dir / "geometry" / "reconstruction"

    depth_anything_dir.mkdir(parents=True, exist_ok=True)
    uni_depth_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Common environment (GPU binding)
    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = gpu

    # 1) DepthAnything
    env_da = env_base.copy()
    da_py_path = str(base_dir / "third_party" / "mega-sam")
    env_da["PYTHONPATH"] = env_da.get("PYTHONPATH", "")
    env_da["PYTHONPATH"] = (env_da["PYTHONPATH"] + os.pathsep if env_da["PYTHONPATH"] else "") + da_py_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "Depth-Anything" / "run_videos.py"),
        "--encoder", "vitl",
        "--load-from", str(base_dir / "third_party" / "mega-sam" / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"),
        "--img-path", str(frame_dir),
        "--outdir", str(depth_anything_dir)
    ], env_da)

    # 2) UniDepth
    env_uni = env_base.copy()
    uni_py_path = str(base_dir / "third_party" / "mega-sam" / "UniDepth")
    env_uni["PYTHONPATH"] = env_uni.get("PYTHONPATH", "")
    env_uni["PYTHONPATH"] = (env_uni["PYTHONPATH"] + os.pathsep if env_uni["PYTHONPATH"] else "") + uni_py_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "UniDepth" / "scripts" / "demo_mega-sam.py"),
        "--img-path", str(frame_dir),
        "--outdir", str(uni_depth_dir)
    ], env_uni)

    # 3) Camera Tracking
    env_track = env_base.copy()
    base_path = str(base_dir / "third_party" / "mega-sam" / "base")
    droid_path = str(base_dir / "third_party" / "mega-sam" / "base" / "droid_slam")
    parts = [p for p in [env_track.get("PYTHONPATH", ""), base_path, droid_path] if p]
    env_track["PYTHONPATH"] = os.pathsep.join(parts)

    ckpt_path = base_dir / "third_party" / "mega-sam" / "checkpoints" / "megasam_final.pth"
    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "camera_tracking_scripts" / "test_demo.py"),
        "--datapath=" + str(frame_dir),
        "--weights=" + str(ckpt_path),
        "--output_path", str(recon_dir),
        "--mono_depth_path", str(depth_anything_dir),
        "--metric_depth_path", str(uni_depth_dir),
        "--disable_vis"
    ], env_track)

    # 4) RAFT Optical Flow
    env_raft = env_base.copy()
    raft_core_path = str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "core")
    env_raft["PYTHONPATH"] = env_raft.get("PYTHONPATH", "")
    env_raft["PYTHONPATH"] = (env_raft["PYTHONPATH"] + os.pathsep if env_raft["PYTHONPATH"] else "") + raft_core_path

    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "preprocess_flow.py"),
        "--datapath=" + str(frame_dir),
        "--model=" + str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "raft-things.pth"),
        "--output_path", str(recon_dir),
        "--mixed_precision"
    ], env_raft)

    # 5) CVD optimization
    env_cvd = env_base.copy()
    run([
        "python",
        str(base_dir / "third_party" / "mega-sam" / "cvd_opt" / "cvd_opt.py"),
        "--output_dir", str(recon_dir),
        "--data_dir", str(recon_dir),
        "--image_dir", str(data_dir / "resized_images"),
        "--w_grad", "2.0",
        "--w_normal", "5.0"
    ], env_cvd)

    print(f"[Visualization] Saving multi-step PCD HTML to: {recon_dir / 'multistep_pcd.html'}")
    from utils.viz_dynamic_pcd import save_multistep_pcd_html
    save_multistep_pcd_html(video_name, max_points=5000)

    print(f"[Done] {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_pipeline")
    parser.add_argument("--video_name", type=str, default=None, help="If set, run only this key")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [lab1, ...]")
    args = parser.parse_args()

    # If a single key is given via CLI, run it; else read keys from YAML and run all.
    if args.video_name is not None:
        main(args.video_name, args.gpu)
    else:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k in cfg["keys"]:
            main(k, args.gpu)
