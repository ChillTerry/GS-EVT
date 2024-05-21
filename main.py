import os
import sys
BASE_DIR = os.getcwd()
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import yaml
import torch
from munch import munchify
from argparse import ArgumentParser

from utils.common import load_events_from_txt
from utils.tracker import Tracker
from utils.render_camera.camera import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel


def main(config_path):
    # Load config
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)

    # Setup parameters
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    event_data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]

    # Setup camera (viewpoint)
    viewpoint = Camera.init_from_yaml(config)

    # Setup gaussian model
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    # Setup event data
    event_arrays = load_events_from_txt(event_data_path, max_events_per_frame)

    # Init tracker
    tracker = Tracker(config, event_arrays, viewpoint, gaussians, pipeline, background)

    # Go tracking
    tracker.tracking()

if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    main(args.config_path)
