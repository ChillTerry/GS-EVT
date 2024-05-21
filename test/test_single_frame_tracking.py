import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
import torch
import numpy as np
from argparse import ArgumentParser

from utils.pose import update_pose
from utils.common import load_events_from_txt, init_gs
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame


def test_single_frame_tracking(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]
    intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
    distortion_factors = np.array(config["Event"]["distortion_factors"])

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    event_arrays = load_events_from_txt(data_path, max_events_per_frame, num_arrays=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, event_arrays[0])
    event_frame = eFrame.event_frame.detach().cpu().numpy().transpose(1, 2, 0)

    view, gaussians, pipeline, background = init_gs(config)
    view.T += 0.2   # Add disturbance on translation
    rFrame1 = RenderFrame(view, gaussians, pipeline, background)
    view.cam_trans_delta.data.add_(0.1)
    view.cam_rot_delta.data.add_(0.01)
    update_pose(view)
    rFrame2 = RenderFrame(view, gaussians, pipeline, background)
    intensity_change_frame = torch.abs(rFrame1.intensity_frame - rFrame2.intensity_frame) * 255
    intensity_change_frame = intensity_change_frame.detach().cpu().numpy().transpose(1, 2, 0)
    merge_img = np.zeros((intensity_change_frame.shape[0], intensity_change_frame.shape[1], 3), dtype=np.uint8)
    merge_img[:, :, 0] = intensity_change_frame.squeeze(axis=-1)
    merge_img[:, :, 2] = event_frame.squeeze(axis=-1)

    cv2.imwrite(os.path.join(results_dir, 'event_intensity_overlay_frame.png'), merge_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_single_frame_tracking(args.config_path)
