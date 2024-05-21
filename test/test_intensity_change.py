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

from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from utils.pose import update_pose
from gaussian_splatting.scene.gaussian_model import GaussianModel


def init(config_path):
    # Setup parameters
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Setup camera viewpoint
    view = Camera.init_from_yaml(config)

    # Setup gaussian model
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    return view, gaussians, pipeline, background


def save_render_image(rFrame: RenderFrame, id=None):
    import torchvision
    from torchvision.transforms.functional import to_pil_image
    if id is not None:
        depth_image_name = f"depth_{id}.png"
        color_image_name = f"color_{id}.png"
    else:
        depth_image_name = f"depth.png"
        color_image_name = f"color.png"

    results_path = os.path.join(BASE_DIR, "results")
    os.makedirs(results_path, exist_ok=True)
    render_image = rFrame.color_frame
    render_depth = rFrame.depth_frame
    # Save images
    min_val = torch.min(render_depth)
    max_val = torch.max(render_depth)
    normalized_depth_tensor = (render_depth - min_val) / (max_val - min_val)
    normalized_depth_tensor = torch.clamp(normalized_depth_tensor, 0, 1)
    depth_image = to_pil_image(normalized_depth_tensor)
    depth_image.save(os.path.join(results_path, depth_image_name))

    torchvision.utils.save_image(render_image, os.path.join(results_path, color_image_name))


def test_intensity_change(config_path):
    view, gaussians, pipeline, background = init(config_path)
    rFrame1 = RenderFrame(view, gaussians, pipeline, background)
    save_render_image(rFrame1, id=1)

    view.cam_trans_delta.data.add_(0.1)
    view.cam_rot_delta.data.add_(0.01)
    update_pose(view)
    rFrame2 = RenderFrame(view, gaussians, pipeline, background)
    save_render_image(rFrame2, id=2)

    intensity_change = torch.abs(rFrame1.intensity_frame - rFrame2.intensity_frame)
    import torchvision
    results_path = os.path.join(BASE_DIR, "results")
    torchvision.utils.save_image(intensity_change, os.path.join(results_path, "intensity_change.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_intensity_change(args.config_path)
