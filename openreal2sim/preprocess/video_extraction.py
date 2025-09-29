#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frames from videos using ffmpeg, reading configs from YAML file.
Additionally:
- Resize extracted frames to match the resolution of data/{video_name}.png or .jpg (if present).
- Save data/{video_name}.png as outputs/{video_name}/images/frame_00000.jpg so it's the first frame.
"""

import os
import subprocess
import yaml
from pathlib import Path
from typing import Tuple

from PIL import Image  # Required for reading size and saving first frame


def get_ref_image_path(video_name: str) -> Path:
    """
    Return the path to the reference image for this video.
    Priority: PNG first, then JPG.
    """
    png_path = Path("data") / f"{video_name}.png"
    if png_path.is_file():
        return png_path
    jpg_path = Path("data") / f"{video_name}.jpg"
    if jpg_path.is_file():
        return jpg_path
    raise FileNotFoundError(f"Reference image not found for video '{video_name}': "
                            f"expected {png_path} or {jpg_path}.")

def read_ref_image_size(video_name: str) -> Tuple[int, int]:
    """
    Return (width, height) of data/{video_name}.png or .jpg.
    Assumes one of them exists.
    """
    ref_img = get_ref_image_path(video_name)
    with Image.open(ref_img) as im:
        return im.width, im.height


def save_ref_image_as_first_frame(video_name: str, output_dir: Path):
    """
    Convert data/{video_name}.png or .jpg to outputs/.../frame_00000.jpg.
    Always store at highest JPEG quality.
    """
    ref_img = get_ref_image_path(video_name)
    output_path = output_dir / "frame_00000.jpg"

    with Image.open(ref_img) as im:
        # Ensure 3-channel RGB; if alpha present, composite over white background
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
            rgb = bg
        else:
            rgb = im.convert("RGB")
        # Highest quality JPEG: quality=100, no chroma subsampling
        rgb.save(output_path, format="JPEG", quality=100, subsampling=0, optimize=True)
    print(f"[Info] Wrote first frame from image: {output_path}")


def run_ffmpeg(video_name: str, fps: int, quality: int):
    """Run ffmpeg to extract frames from one video, scaling to reference image size."""
    input_path = Path("data") / f"{video_name}.mp4"
    output_dir = Path("outputs") / video_name / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect reference image size (PNG or JPG); assume exists
    width, height = read_ref_image_size(video_name)

    # Build ffmpeg filter chain: fps + scale to match the reference image
    vf = f"fps={fps},scale={width}:{height}:flags=lanczos"

    # ffmpeg command:
    # -start_number 1 to leave frame_00000.jpg for the reference image we add separately.
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without prompt
        "-i", str(input_path),
        "-vf", vf,
        "-q:v", str(quality),          # 1(best)..31(worst) for mjpeg
        "-start_number", "1",          # start from frame_00001.jpg
        str(output_dir / "frame_%05d.jpg"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "LD_LIBRARY_PATH": ""})

    # After extraction, save the reference image as frame_00000.jpg (highest quality)
    save_ref_image_as_first_frame(video_name, output_dir)


def main(config_file: str = "config/config.yaml"):
    """Main function: load config and process videos."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    fps = config["preprocess"]["video_extraction"].get("fps", 25)
    quality = config["preprocess"]["video_extraction"].get("quality", 1)
    videos = config["keys"]

    for video_name in videos:
        run_ffmpeg(video_name, fps, quality)

if __name__ == "__main__":
    main()
