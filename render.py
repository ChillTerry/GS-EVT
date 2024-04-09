import os
import sys
BASE_DIR = os.getcwd()
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import yaml
from munch import munchify
from argparse import ArgumentParser
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image

from utils.rendering_camera.camera import Camera
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import searchForMaxIteration


def my_render(config):
    # Setup parameters
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline_params = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    results_path = "./results"
    os.makedirs(results_path, exist_ok=True)

    # Setup camera viewpoint
    view = Camera.init_from_yaml(config)

    # Setup gaussian model
    gaussians = GaussianModel(model_params.sh_degree)
    pc_path = os.path.join(model_params.model_path, "point_cloud")
    loaded_iter = searchForMaxIteration(pc_path)
    gaussians.load_ply(os.path.join(model_params.model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"))

    # Start rendering
    rendering = render(view, gaussians, pipeline_params, background)
    rendering_image = rendering["render"]
    rendering_depth = rendering["depth"]

    # Save images
    min_val = torch.min(rendering_depth)
    max_val = torch.max(rendering_depth)
    normalized_depth_tensor = (rendering_depth - min_val) / (max_val - min_val)
    normalized_depth_tensor = torch.clamp(normalized_depth_tensor, 0, 1)
    depth_image = to_pil_image(normalized_depth_tensor)
    depth_image.save(os.path.join(results_path, "rendering_depth.png"))

    torchvision.utils.save_image(rendering_image, os.path.join(results_path, "rendering_image.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    with open(args.config_path, "r") as yml:
        config = yaml.safe_load(yml)

    my_render(config)