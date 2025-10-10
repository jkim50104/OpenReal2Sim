# IsaacLab

## Installation

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then, launch the container:
```
docker compose -p "$USER" -f docker/compose.yml up -d isaaclab 
```
and enter it:
```
docker compose -p "$USER" -f docker/compose.yml exec isaaclab bash
```