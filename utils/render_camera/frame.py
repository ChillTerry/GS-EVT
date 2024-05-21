import torch
import torch.nn.functional as F

from utils.render_camera.camera import Camera
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel

class RenderFrame:
    """
    Frame that rendered by gaussian splatting
    """

    def __init__(self, viewpoint: Camera, gaussians: GaussianModel, pipeline, background):
        self.viewpoint = viewpoint
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background

        rendering = render(viewpoint, gaussians, pipeline, background)
        self.color_frame = rendering["render"]
        self.depth_frame = rendering["depth"]

        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        weights = weights.to(self.color_frame.device)
        self.intensity_frame = (self.color_frame * weights).sum(dim=1)

    def get_grad_frame(self):
        sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                    [-2., 0., 2.],
                                    [-1., 0., 1.]]).view((1, 1, 3, 3))
        sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                    [0., 0., 0.],
                                    [1., 2., 1.]]).view((1, 1, 3, 3))

        if self.intensity_frame.is_cuda:
            sobel_x_kernel = sobel_x_kernel.cuda()
            sobel_y_kernel = sobel_y_kernel.cuda()

        sobel_x = F.conv2d(self.intensity_frame, sobel_x_kernel, padding=1)
        sobel_y = F.conv2d(self.intensity_frame, sobel_y_kernel, padding=1)
        self.grad_frame = torch.sqrt(sobel_x**2 + sobel_y**2)

    def get_intensity_change_frame(self):
        pass