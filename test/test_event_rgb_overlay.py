import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
import numpy as np
from argparse import ArgumentParser

from test_intensity_change import init
from utils.common import load_events_from_txt
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame


def test_event_rgb_overlay(config_path):
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
    view, gaussians, pipeline, background = init(config_path)
    rFrame = RenderFrame(view, gaussians, pipeline, background)

    out_img = rFrame.color_frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_img[..., 2] = eFrame.event_frame.cpu().numpy().squeeze(axis=-1)
    print(eFrame.event_frame.shape)

    # import torchvision
    # torchvision.utils.save_image(rFrame.color_frame, os.path.join(results_dir, "test.png"))
    cv2.imwrite(os.path.join(results_dir, 'event_rgb_overlay_frame.png'), out_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_event_rgb_overlay(args.config_path)
