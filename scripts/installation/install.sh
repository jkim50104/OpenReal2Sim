#!/usr/bin/env bash
set -euo pipefail

JOBS="${MAX_JOBS:-8}"

# mega-sam
mkdir -p $REPO_ROOT/third_party/mega-sam/Depth-Anything/checkpoints
wget -O $REPO_ROOT/third_party/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O $REPO_ROOT/third_party/mega-sam/cvd_opt/raft-things.pth

# segmentation
wget -O $REPO_ROOT/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# segmentation cuda extension
cd $REPO_ROOT/third_party/Grounded-SAM-2 && \
  python build_cuda.py build_ext --inplace -v && \
  cd $REPO_ROOT

# foundation pose
mkdir -p $REPO_ROOT/third_party/FoundationPose/weights
gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O $REPO_ROOT/third_party/FoundationPose/weights

# foundation pose compile
cd $REPO_ROOT/third_party/FoundationPose/mycpp && \
  rm -rf build && \
  mkdir -p build && cd build && \
  cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j"${JOBS}" && \
  cd $REPO_ROOT

cd $REPO_ROOT/third_party/FoundationPose/bundlesdf/mycuda && \
  rm -rf build *egg* && \
  python -m pip install . --no-build-isolation && \
  cd $REPO_ROOT

# Wilor
wget -O $REPO_ROOT/third_party/WiLoR/pretrained_models/wilor_final.ckpt \
   wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt
wget -O $REPO_ROOT/third_party/WiLoR/pretrained_models/detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
  
# GraspGen
cd $REPO_ROOT/third_party/GraspGen && \
  git clone https://huggingface.co/adithyamurali/GraspGenModels && \
  cd GraspGenModels && \
  git lfs install && \
  git lfs pull && \
  cd $REPO_ROOT