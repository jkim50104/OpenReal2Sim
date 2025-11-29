FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv python-is-python3 \
    python3-distutils \
    libopenblas-dev \
    build-essential cmake ninja-build pkg-config \
    git curl wget ffmpeg bash libspatialindex-dev \
    ncurses-bin ncurses-term \
    libglu1-mesa libxkbcommon-x11-0 libegl1 libgl1 libgles2 \
    libxinerama1 libxcursor1 libsm6 libice6 libxrender1 libxrandr2 \
    libxt6 libxmu6 \
    && python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools packaging \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
SHELL ["/bin/bash", "-lc"]

ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1


RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128


RUN sed -i 's|auto __raw = __to_address(__r.get())|auto __raw = std::__to_address(__r.get())|' \
    /usr/include/c++/11/bits/shared_ptr_base.h


ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
# curobo
RUN echo $HOME
RUN git clone https://github.com/NVlabs/curobo.git "$HOME/curobo" && \
    cd "$HOME/curobo" && pip install -e . --no-build-isolation --user

# Isaac Sim 4.5.0
RUN pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Revert the fixed pytorch version
ENV TERM=xterm-256color
RUN git clone https://github.com/PointsCoder/IsaacLab.git "$HOME/IsaacLab" && \
    cd "$HOME/IsaacLab" && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab/setup.py && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab_tasks/setup.py && \
    sed -i 's|torch==2.5.1|torch==2.7.0|' source/isaaclab_rl/setup.py && \
    ./isaaclab.sh --install

# usd conversion 
# curobo changes the numpy version from 1.23.4 to 2.x, need to get it back
RUN pip install loguru rich numpy==1.23.4

ENV PYTHONPATH=/app:$HOME/third_party
