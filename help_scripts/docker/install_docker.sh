# Use Pre-Built Docker Image
docker pull ghcr.io/pointscoder/openreal2sim:dev
docker tag ghcr.io/pointscoder/openreal2sim:dev openreal2sim:dev

# Run the real2sim container
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml run --publish 8000:5000 openreal2sim

#Inside the docker container, run the following script to download pretrained checkpoints and compile c++/cuda extensions:
python scripts/installation/install.py

# Execute a script inside the container
# Inside the container’s terminal and from the repository root:
python <path_to_script>.py

# Run the isaaclab container
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml run isaaclab