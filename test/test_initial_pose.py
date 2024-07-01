import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import yaml
import torch
import torchvision
from munch import munchify
from argparse import ArgumentParser

from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel


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
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background)
    delta_Ir = rFrame.render()["render"]

    results_path = os.path.join(BASE_DIR, "results")
    torchvision.utils.save_image(delta_Ir, os.path.join(results_path, "initial_view.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/VECTOR/desk_normal1_config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_intensity_change(args.config_path)
