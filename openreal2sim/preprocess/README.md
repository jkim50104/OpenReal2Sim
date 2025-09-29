# Data Preprocessing

This directory contains scripts and instructions for preprocessing raw images or videos into depth maps, camera intrinsics/extrinsics, and dynamic point clouds (a.k.a 4D reconstruction).

These information recovers the underlying geometry and motion of the scene, which can be further used in the `reconstruction` step to construct a physically interactable scene.

## Code Structure

```
openreal2sim/
├── preprocess
│   ├── utils
│   │   ├── depth_to_color.py        # Utility functions for rendering colored depth maps
│   │   ├── viz_dynamic_pcd.py       # Utility functions for visualizing dynamic point clouds (as a quality check of the preprocessing step)
│   ├── video_extraction.py          # Extract frames from videos using ffmpeg
│   ├── video_depth_estimation.py    # Running 4D reconstruction to obtain depth, camera infos, and dynamic point clouds from videos
│   ├── image_depth_estimation.py    # Running single-image depth estimation to obtain depth and camera intrinsics from images
│   ├── video_depth_alignment.py     # (Optional) if we have GT depth map, e.g., depth image from an RGB-D camera, align the estimated video depth to GT depth
│   └── README.md                    # This file
```