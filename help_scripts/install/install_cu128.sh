#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Config (override via env)
# ---------------------------
ENV_NAME="${ENV_NAME:-openreal2sim}"
PY_VER="${PY_VER:-3.11}"
MAX_JOBS="${MAX_JOBS:-64}"

# RTX PRO 6000 Blackwell = SM120
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

REQ_FILE="${REQ_FILE:-docker/real2sim/requirements.docker.txt}"
REPO_ROOT="$(pwd)"

# ---------------------------
# Preflight
# ---------------------------
if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: requirements file not found: $REQ_FILE"
  echo "Run this from your repo root, or set REQ_FILE=/abs/path/to/requirements.docker.txt"
  exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
  echo "ERROR: micromamba not found in PATH."
  exit 1
fi

# Initialize micromamba shell (for 'micromamba activate' in scripts)
if micromamba shell hook --help >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell=bash)"
else
  eval "$(micromamba shell hook -s bash)"
fi

# ---------------------------
# Create env (if missing) + activate
# ---------------------------
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  micromamba create -y -n "$ENV_NAME" \
    -c conda-forge -c nvidia --strict-channel-priority \
    "python=${PY_VER}" pip
fi

micromamba activate "$ENV_NAME"

# ---------------------------
# Pin pip / setuptools / wheel FIRST
# ---------------------------
python -m pip install --no-cache-dir \
  "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1" packaging

# ---------------------------
# Conda(-forge) build/runtime deps (Docker apt equivalent-ish)
# ---------------------------
micromamba install -y -n "$ENV_NAME" -c conda-forge --strict-channel-priority \
  cmake ninja pkg-config git curl wget ffmpeg make \
  c-compiler cxx-compiler \
  openblas \
  boost-cpp eigen pybind11 \
  libspatialindex

# ---------------------------
# If nvcc missing -> install CUDA 12.8.1 (your requested command, micromamba version)
# ---------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[INFO] nvcc not found; installing CUDA 12.8.1 into the env..."
  # micromamba equivalent of:
  # conda install nvidia/label/cuda-12.8.1::cuda -c nvidia/label/cuda-12.8.1 -y
  micromamba install -y -n "$ENV_NAME" \
    -c nvidia/label/cuda-12.8.1 \
    nvidia/label/cuda-12.8.1::cuda
fi

# ---------------------------
# PyTorch 2.7.1 + cu128
# ---------------------------
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# ---------------------------
# Export build env BEFORE compiling anything
# ---------------------------
export MAX_JOBS="${MAX_JOBS}"
export NINJAFLAGS="-j${MAX_JOBS}"
export MAKEFLAGS="-j${MAX_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Best-effort CUDA_HOME detection (for builds)
CUDA_HOME_DETECTED=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH="$(command -v nvcc)"
  if [[ "$NVCC_PATH" == "$CONDA_PREFIX"* ]]; then
    CUDA_HOME_DETECTED="$CONDA_PREFIX"
  else
    CUDA_HOME_DETECTED="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
  fi
elif [[ -d /usr/local/cuda ]]; then
  CUDA_HOME_DETECTED="/usr/local/cuda"
fi
if [[ -n "$CUDA_HOME_DETECTED" ]]; then
  export CUDA_HOME="$CUDA_HOME_DETECTED"
fi

# ---------------------------
# xFormers (force source build for SM120)
# ---------------------------
export PIP_NO_BINARY="xformers"
export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.31.post1" #"0.0.31.post1" nystrom only supports up to 0.0.29 (UniDepth uses it)
pip install -v --no-build-isolation \
    git+https://github.com/facebookresearch/xformers.git@v${SETUPTOOLS_SCM_PRETEND_VERSION}#egg=xformers
unset PIP_NO_BINARY

# ---------------------------
# torch-scatter (PyG wheels for torch 2.7.1 + cu128)
# ---------------------------
python -m pip install --no-cache-dir torch-scatter \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

# ---------------------------
# Rest of requirements.docker.txt
# ---------------------------
python -m pip install --no-cache-dir -r "$REQ_FILE"

# ---------------------------
# Third-party compiled deps (only if folders exist)
# ---------------------------
if [[ -d "$REPO_ROOT/third_party/mega-sam/base" ]]; then
  pushd "$REPO_ROOT/third_party/mega-sam/base" >/dev/null
  python setup.py install
  popd >/dev/null
fi

if [[ -d "$REPO_ROOT/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer" ]]; then
  pushd "$REPO_ROOT/third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer" >/dev/null
  python setup.py install
  popd >/dev/null
fi

if [[ -d "$REPO_ROOT/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer" ]]; then
  pushd "$REPO_ROOT/third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer" >/dev/null
  python setup.py install
  popd >/dev/null
fi

python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVlabs/nvdiffrast.git@abb07ca0358f3a21c3942b50c54aa1eacd329af9"

python -m pip install --no-cache-dir kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html

PIP_NO_BUILD_ISOLATION=1 python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/facebookresearch/pytorch3d.git"

if [[ -d "$REPO_ROOT/third_party/GraspGen" ]]; then
  pushd "$REPO_ROOT/third_party/GraspGen" >/dev/null
  python -m pip install -e .
  popd >/dev/null
fi

if [[ -d "$REPO_ROOT/third_party/GraspGen/pointnet2_ops" ]]; then
  pushd "$REPO_ROOT/third_party/GraspGen/pointnet2_ops" >/dev/null
  python -m pip install --no-build-isolation .
  popd >/dev/null
fi

python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/mattloper/chumpy"

# ---------------------------
# Mirror Docker ENV (activate hook)
# ---------------------------
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/real2sim_env.sh" <<EOF
export MAX_JOBS=${MAX_JOBS}
export MAKEFLAGS="-j${MAX_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
${CUDA_HOME_DETECTED:+export CUDA_HOME="${CUDA_HOME_DETECTED}"}
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
EOF
# export TORCH_EXTENSIONS_DIR="/tmp/.cache/torch_extensions"
# export NUMBA_CACHE_DIR="/tmp/numba_cache"
# export HF_HOME="${REPO_ROOT}/.cache/huggingface"

# mkdir -p /tmp/.cache/torch_extensions /tmp/numba_cache "$REPO_ROOT/.cache/huggingface" || true

# ---------------------------
# Sanity check
# ---------------------------
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
try:
    import xformers
    print("xformers:", xformers.__version__)
except Exception as e:
    print("xformers import failed:", e)
PY

echo
echo "✅ Done."
echo "Activate with: micromamba activate ${ENV_NAME}"


# ------------------------------------------------------------
# Run project installer (downloads + building third_party)
# ------------------------------------------------------------
if [[ -f "$REPO_ROOT/scripts/installation/install.sh" ]]; then
  echo
  echo "🚀 Running project installer: $REPO_ROOT/scripts/installation/install.sh"
  (cd "$REPO_ROOT" && bash "$REPO_ROOT/scripts/installation/install.sh")
else
  echo
  echo "⚠️  Skipping: $REPO_ROOT/scripts/installation/install.sh not found"
fi

# Install latest Unidepth due to xformers incompatability
git clone https://github.com/lpiccinelli-eth/UniDepth.git