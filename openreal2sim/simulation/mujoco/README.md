# MuJoCo Simulation

This module provides tools for converting reconstructed 3D assets to MuJoCo simulation format and fusing them with robot models.

## Installation

### Docker Installation

Build the MuJoCo container locally:

```bash
docker build -t mujoco:dev -f docker/mujoco/Dockerfile .
```

Or using docker compose:

```bash
docker compose -f docker/compose.yml build mujoco
```

**Before launching**, enable X11 forwarding on the host:

```bash
xhost +local:
```

Launch the container:

```bash
docker compose -f docker/compose.yml up -d mujoco
docker compose -f docker/compose.yml exec mujoco bash
```

### Virtual Environment Installation

Alternatively, install dependencies in a virtual environment:

```bash
pip install -r docker/mujoco/requirements.docker.txt
```

Then run the tools under `openreal2sim/simulation/mujoco/tools` directly.

## Workflow

### (Optional) Simplify GLB Meshes

```bash
python openreal2sim/simulation/mujoco/tools/simplify_scene.py --scene-name demo_genvideo
```

**WARNING: This overwrites GLB files in place. Make sure you have backups!**

### GLB to MJCF Conversion

Convert reconstructed GLB meshes to MuJoCo MJCF format. This script is adapted from [obj2mjcf](https://github.com/kevinzakka/obj2mjcf).

**For reconstructed scenes:**

```bash
python openreal2sim/simulation/mujoco/tools/glb_to_mjcf.py --scene-name demo_genvideo
```

This will:
- Parse `scene.json` to identify objects and background
- Convert each GLB to MJCF using CoACD for convex decomposition
- Output MJCF assets to `outputs/<scene_name>/simulation/mujoco/mjcf/`


### Scene Fusion

Fuse MJCF assets with the Franka Panda robot model.

```bash
python openreal2sim/simulation/mujoco/tools/fuse_scene.py --scene-name demo_genvideo
```

This will:
- Load MJCF assets from `outputs/<scene_name>/simulation/mujoco/mjcf/`
- Compute robot pose based on heuristics (or use provided pose)
- Fuse robot and scene together
- Output complete scene to `outputs/<scene_name>/simulation/mujoco/scene.xml`

**Note:** This uses XML manipulation to merge scene elements. Complex scenes may require manual adjustments.

Default simulation parameters are defined in `config/constants.yaml`. You can modify this file to adjust collision margins, solver settings, material properties, etc. You can also specify a custom constants file using `--constants-path`.


**Additional Options:**

- `--default-mass`: Default object mass when not in config (default: 0.1 kg)
- `--object-masses`: Path to YAML file with object masses
- `--z-offset`: Vertical offset for objects (default: 0.005)
- `--inertia-scale`: Inertia scaling factor (default: 0.002)
- `--groundplane-height`: Height of ground plane (default: 0.0)

**Custom Object Masses:**

Create a YAML file (e.g., `masses.yaml`):

```yaml
plate: 0.15
spoon: 0.05
pen: 0.02
```

Use it:

```bash
python openreal2sim/simulation/mujoco/tools/fuse_scene.py \
  --scene-name demo_genvideo \
  --object-masses masses.yaml
```

Objects not in the file will use `--default-mass` with a warning.

### Visualization

View the fused scene with MuJoCo's viewer:

```bash
python -m mujoco.viewer --mjcf outputs/demo_genvideo/simulation/mujoco/scene.xml
```

### Trajectory Replay

Replay trajectories generated from IsaacSim heuristics:

```bash
python openreal2sim/simulation/mujoco/tools/replay_trajectory.py \
  --demo-path outputs/demo_genvideo/demos/demo_0/env_000
```

This script:
- Loads robot joint trajectories from `joint_pos.npy`, `joint_vel.npy`, and `gripper_cmd.npy`
- Initializes objects at their reconstructed positions from `config.json`
- Uses PD control with gravity compensation for realistic physics
- Provides interactive viewer with keyboard controls (Space: pause, R: restart)

**Options:**
- `--loop`: Loop trajectory playback continuously
- `--default-mass`: Default object mass in kg (default: 0.1)
- `--object-masses`: Path to YAML file with custom object masses


Due to dynamics gap between simulators, trajectories that work in Isaac Sim may not always succeed in MuJoCo. You may need to adjust simulation parameters to achieve stable playback.

## Output Structure

After running the pipeline, you'll have:

```
outputs/<scene_name>/simulation/
├── mujoco/
│   ├── mjcf/
│   │   ├── background_registered/
│   │   │   ├── background_registered.xml
│   │   │   ├── background_registered.obj
│   │   │   ├── background_registered_collision_*.obj
│   │   │   └── material_0.png
│   │   ├── 1_object_optimized/
│   │   │   ├── 1_object_optimized.xml
│   │   │   ├── 1_object_optimized.obj
│   │   │   ├── 1_object_optimized_collision_*.obj
│   │   │   └── material_0.png
│   │   └── ...
│   └── scene.xml  # Fused scene ready for MuJoCo
├── scene.json
├── background_registered.glb
└── *_optimized.glb
```

## Configuration

- **Simulation parameters**: `config/constants.yaml` contains default collision, solver, and material settings
- **Robot configuration**: `config/franka_panda_config.yaml` contains joint names, PD gains, and gripper settings

## License

The GLB to MJCF conversion code is adapted from [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) (MIT License).

