import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
import torch
import numpy as np
from munch import munchify
from argparse import ArgumentParser

from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.event_camera.event import EventFrame, load_events_from_txt


def test_event_rgb_overlay(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    device = config["Gaussian"]["model_params"]["device"]
    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]
    intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
    distortion_factors = np.array(config["Event"]["distortion_factors"])
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor(config["Gaussian"]["background"], dtype=torch.float32, device=device)
    viewpoint = Camera.init_from_yaml(config)
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    event_arrays = load_events_from_txt(data_path, max_events_per_frame, num_arrays=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, event_arrays[0])
    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background)

    out_img = rFrame.intensity_frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_img[..., 2] = eFrame.delta_Ie.detach().cpu().numpy().squeeze(axis=0) * 255

    cv2.imwrite(os.path.join(results_dir, 'event_rgb_overlay_frame.png'), out_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_event_rgb_overlay(args.config_path)
