import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import yaml
import torch
from munch import munchify
from argparse import ArgumentParser

from utils.pose import update_pose
from utils.common import init_gs, save_render_image
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel


def test_intensity_change(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    view, gaussians, pipeline, background = init_gs(config)
    rFrame1 = RenderFrame(view, gaussians, pipeline, background)
    # save_render_image(rFrame1, id=1)

    view.cam_trans_delta.data.add_(0.1)
    view.cam_rot_delta.data.add_(0.01)
    update_pose(view)
    rFrame2 = RenderFrame(view, gaussians, pipeline, background)
    # save_render_image(rFrame2, id=2)

    intensity_change = torch.abs(rFrame1.intensity_frame - rFrame2.intensity_frame)
    import torchvision
    results_path = os.path.join(BASE_DIR, "results")
    torchvision.utils.save_image(intensity_change, os.path.join(results_path, "intensity_change.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_intensity_change(args.config_path)
