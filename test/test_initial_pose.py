import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import yaml
import torch
import torchvision
import numpy as np
from munch import munchify
from argparse import ArgumentParser
import cv2

from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.event_camera.event import EventFrame
from utils.event_camera.event import load_events_from_txt

def render_frame(data_path, img_width, img_height, intrinsic, distortion_factors, max_events_per_frame, filter_threshold):
    event_arrays = load_events_from_txt(data_path, max_events_per_frame, array_nums=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, filter_threshold, event_arrays[0])
    delta_Ie_np = eFrame.delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
    color_Ie = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
    negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
    positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
    color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
    color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)

    color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)
    return color_Ie

def test_intensity_change(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)

    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    device = model_params.device
    background = torch.tensor(model_params.background, dtype=torch.float32, device=device)
    event_data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]

    viewpoint = Camera.init_from_yaml(config)
    viewpoint.delta_tau = 0.1
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background, 3)
    # delta_Ir = rFrame.render()["render"]
    delta_Ir = rFrame.intensity_frame
    color_Ir = rFrame.plot(delta_Ir)

    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]
    intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
    distortion_factors = np.array(config["Event"]["distortion_factors"])
    filter_threshold = config["Event"]["filter_threshold"]
    
    event_arrays = load_events_from_txt(data_path, max_events_per_frame, array_nums=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, filter_threshold, event_arrays[0])
    color_Ie = eFrame.plot(eFrame.pose_Ie)
    print(color_Ie.dtype, color_Ir.dtype)

    alpha = 0.8  # Define the transparency level: 0.0 - completely transparent; 1.0 - completely opaque
    img = cv2.addWeighted(color_Ie, alpha, color_Ir, 1 - alpha, 0)
    
    cv2.imwrite(os.path.join("./results", f'overlay_img.png'), img)

if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    # parser.add_argument("--config_path", "-c", type=str, default="./configs/VECTOR/desk_normal1_config.yaml")
    # parser.add_argument("--config_path", "-c", type=str, default="./configs/VECTOR/robot_normal1_config.yaml")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/VECTOR/sofa_normal1_config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_intensity_change(args.config_path)
