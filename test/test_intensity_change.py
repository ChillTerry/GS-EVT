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
    # from torchvision import transforms
    # from PIL import Image

    # # Define the transformation to convert the image to a tensor
    # transform = transforms.ToTensor()

    # # Load the image with PIL
    # img1_path = '/home/liutao/Project/GS-EVT/data/table/rgb/1714108580952518.png'
    # img2_path = '/home/liutao/Project/GS-EVT/data/table/rgb/1714108581010107.png'
    # image1 = Image.open(img1_path)
    # image2 = Image.open(img2_path)

    # # # Check if the image needs to be converted to RGB mode (necessary if the image is in palette mode or other modes like L, 1, etc.)
    # # if image1.mode != 'RGB':
    # #     image1 = image1.convert('RGB')

    # # Apply the transform to the image
    # image1_tensor = transform(image1)
    # image2_tensor = transform(image2)
    # intensity_change = torch.abs(image2_tensor - image1_tensor)

    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    device = config["Gaussian"]["model_params"]["device"]
    linear_vel = torch.tensor(config["Tracking"]["initial_vel"]["linear_vel"], device=device, dtype=torch.float32)
    angular_vel = torch.tensor(config["Tracking"]["initial_vel"]["angular_vel"], device=device, dtype=torch.float32)
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor(config["Gaussian"]["background"], dtype=torch.float32, device=device)
    viewpoint = Camera.init_from_yaml(config)
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    delta_tau = 0.172009
    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background)
    delta_Ir = rFrame.get_delta_Ir(delta_tau, linear_vel, angular_vel)
    results_path = os.path.join(BASE_DIR, "results")
    torchvision.utils.save_image(delta_Ir, os.path.join(results_path, "intensity_change.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_intensity_change(args.config_path)
