import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import cv2
import yaml
import torch
import numpy as np
from munch import munchify
from argparse import ArgumentParser

from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render1


def test_depth_rendering(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    device = config["Gaussian"]["model_params"]["device"]
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor(model_params.background, dtype=torch.float32, device=device)
    viewpoint = Camera.init_from_yaml(config)
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)
    results_path = os.path.join(BASE_DIR, "results")

    rendering = render1(viewpoint, gaussians, background)
    depth_frame = rendering["depth"]
    depth_image = depth_frame.squeeze(0).detach().cpu().numpy()

    normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized[np.isinf(normalized) | np.isnan(normalized)] = 0
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(results_path, "depth.png"), heatmap)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_depth_rendering(args.config_path)
