import os
import glob
import torch
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import pickle
import cv2
import yaml

from modules.utils.compose_config import compose_configs

class ReconAgent:
    def __init__(self):
        print('[Info] Initializing ReconAgent...')
        self.base_dir = Path.cwd()
        cfg_path = self.base_dir / "config" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.open("r"))
        self.keys = cfg["keys"]
        self.key_cfgs = [compose_configs(key, cfg) for key in self.keys]
        self.key_scene_dicts = {}
        for key in self.keys:
            scene_pkl = self.base_dir / f'outputs/{key}/scene/scene.pkl'
            with open(scene_pkl, 'rb') as f:
                scene_dict = pickle.load(f)
            self.key_scene_dicts[key] = scene_dict
        print('[Info] ReconAgent initialized.')
    
    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def scene_inpainting(self):
        from modules.scene_inpainting import scene_inpainting
        self.key_scene_dicts = scene_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scene inpainting completed.')

    def run(self):
        self.scene_inpainting()
        print('[Info] ReconAgent run completed.')
        return self.key_scene_dicts

if __name__ == '__main__':
    agent = ReconAgent()
    scene_dicts = agent.run()