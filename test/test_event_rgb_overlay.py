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
    background = torch.tensor(model_params.background, dtype=torch.float32, device=device)
    viewpoint = Camera.init_from_yaml(config)
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    event_arrays = load_events_from_txt(data_path, max_events_per_frame, array_nums=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, event_arrays[0])
    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background)
    render_pkg = rFrame.render()
    color_frame = render_pkg["render"]
    color_frame = color_frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    color_frame = color_frame.astype(np.uint8)

    delta_Ie = eFrame.delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
    color_Ie = np.zeros((delta_Ie.shape[0], delta_Ie.shape[1], 3), dtype=np.uint8)
    negative_delta_Ie = np.where(delta_Ie < 0, delta_Ie, 0)
    positive_delta_Ie = np.where(delta_Ie > 0, delta_Ie, 0)
    color_Ie[:, :, 2] = positive_delta_Ie.squeeze(axis=-1)
    color_Ie[:, :, 0] = -negative_delta_Ie.squeeze(axis=-1)

    delta_Ir = rFrame.get_delta_Ir(0.172).detach().cpu().numpy().transpose(1, 2, 0) * 255
    gray_Ir = ((delta_Ir + 255) / 2)
    gray_Ir = gray_Ir.astype(np.uint8)
    gray_Ir = cv2.cvtColor(gray_Ir, cv2.COLOR_GRAY2BGR)

    print(color_frame.shape)
    print(color_Ie.shape)
    print(color_frame.dtype)
    print(color_Ie.dtype)
    cv2.imwrite(os.path.join(results_dir, 'color_frame.png'), color_frame)
    cv2.imwrite(os.path.join(results_dir, 'delta_Ie.png'), color_Ie)
    cv2.imwrite(os.path.join(results_dir, 'delta_Ir.png'), delta_Ir)
    cv2.imwrite(os.path.join(results_dir, 'gray_Ir.png'), gray_Ir)

    # Overlay the color image onto the grayscale image using a weighted sum
    alpha = 0.5  # Define the transparency level: 0.0 - completely transparent; 1.0 - completely opaque
    overlay_img = cv2.addWeighted(color_Ie, alpha, color_frame, 1 - alpha, 0)
    cv2.imwrite(os.path.join(results_dir, 'event_rgb_overlay_frame.png'), overlay_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_event_rgb_overlay(args.config_path)
