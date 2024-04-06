import os
import sys
BASE_DIR = os.getcwd()
GS_PATH = os.path.join(BASE_DIR, "gaussian_splatting")
sys.path.append(GS_PATH)
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image

from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.scene import Scene



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)
    safe_state(silent=False)
    dataset = model.extract(args)
    iteration = -1  # automatically search .ply for max iteration
    results_path = "./results"
    os.makedirs(results_path, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        view = scene.getTrainCameras()[10]
        rendering = render(view, gaussians, pipeline, background)
        rendering_image = rendering["render"]
        rendering_depth = rendering["depth"]
        # print(rendering_image.shape)
        # print(rendering_depth.shape)

        min_val = torch.min(rendering_depth)
        max_val = torch.max(rendering_depth)
        normalized_depth_tensor = (rendering_depth - min_val) / (max_val - min_val)
        normalized_depth_tensor = torch.clamp(normalized_depth_tensor, 0, 1)
        depth_image = to_pil_image(normalized_depth_tensor)
        depth_image.save('rendering_depth.png')

        torchvision.utils.save_image(rendering_image, os.path.join(results_path, "rendering_image.png"))