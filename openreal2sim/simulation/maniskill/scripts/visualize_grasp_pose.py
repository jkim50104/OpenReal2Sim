import gymnasium as gym
import numpy as np
import sapien
import argparse
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OpenReal2SimEnv
from motion.grasp import (
    load_grasps_from_path,
    get_best_grasp_pose,
    transform_grasp_pose,
)
from motion.planner import build_two_finger_gripper_grasp_pose_visual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="/home/haoyang/project/haoyang/OpenReal2Sim/outputs/demo_video/scene/scene.json",
    )
    args = parser.parse_args()

    env = OpenReal2SimEnv(
        scene_json_path=args.scene,
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()

    # --- Grasp Loading and Transformation ---
    target_object_id = next(iter(env.unwrapped.scene_config.objects))
    target_object = env.unwrapped.object_actors[target_object_id]
    grasp_path = env.unwrapped.scene_config.objects[target_object_id].grasps

    grasps = load_grasps_from_path(grasp_path)  # (N, 17)
    best_grasp_local = get_best_grasp_pose(grasps)  # (1, 17)
    import ipdb

    ipdb.set_trace()

    if best_grasp_local is None:
        print(f"No valid grasps found for object {target_object_id}")
        env.close()
        return

    # --- Visualization ---
    print("Visualizing the best grasp pose. Press Ctrl+C to exit.")
    grasp_visual = build_two_finger_gripper_grasp_pose_visual(env.unwrapped.scene)

    # transform the grasp pose to the world frame
    grasp_pose_world = transform_grasp_pose(target_object.pose, best_grasp_local)
    grasp_visual.set_pose(grasp_pose_world)

    try:
        while True:
            env.render()
            env.step(np.zeros(env.action_space.shape))
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExiting.")

    env.close()


if __name__ == "__main__":
    main()
