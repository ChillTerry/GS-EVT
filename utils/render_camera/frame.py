import torch
import torch.nn.functional as F

from utils.render_camera.camera import Camera
from gaussian_splatting.gaussian_renderer import render1, render2
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import focal2fov


class RenderFrame:
    """
    Frame that rendered by gaussian splatting
    """

    def __init__(self, viewpoint: Camera, gaussians: GaussianModel, pipeline, background, pyramid_level):
        self.viewpoint = viewpoint
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.sign_delta_Ir, self.unsign_delta_Ir = self.get_delta_Ir(pyramid_level)

    @property
    def intensity_frame(self):
        rendering = render1(self.viewpoint, self.gaussians, self.background)
        color_frame = rendering["render"]
        intensity_frame = self.get_intensity_frame(color_frame)
        return intensity_frame

    def render(self):
        render_pkg = render1(self.viewpoint, self.gaussians, self.background)
        return render_pkg

    def get_intensity_frame(self, color_frame):
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        weights = weights.to(color_frame.device)
        intensity_frame = (color_frame * weights).sum(dim=1)
        return intensity_frame

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

    def get_delta_Ir(self, pyramid_level=0):
        assert self.viewpoint.delta_tau != 0, f"delta_tau should not be zero, \
        it should be same as the time interval of event frame"
        scale_factor = 0.5 ** pyramid_level
        new_w = int(self.viewpoint.image_width * scale_factor)
        new_h = int(self.viewpoint.image_height * scale_factor)

        last_viewpoint_pose = self.viewpoint.last_vel_transform @ self.viewpoint.curr_pose
        next_viewpoint_pose = self.viewpoint.next_vel_transform @ self.viewpoint.curr_pose
        last_viewpoint_R = last_viewpoint_pose[:3, :3]
        last_viewpoint_t = last_viewpoint_pose[:3, 3]
        next_viewpoint_R = next_viewpoint_pose[:3, :3]
        next_viewpoint_t = next_viewpoint_pose[:3, 3]

        last_viewpoint = Camera(last_viewpoint_R, last_viewpoint_t, self.viewpoint.angular_vel,
                                self.viewpoint.linear_vel, focal2fov(self.viewpoint.fx * scale_factor, new_w),
                                focal2fov(self.viewpoint.fy * scale_factor, new_h),
                                new_w, new_h, delta_tau=self.viewpoint.delta_tau)
        next_viewpoint = Camera(next_viewpoint_R, next_viewpoint_t, self.viewpoint.angular_vel,
                                self.viewpoint.linear_vel, focal2fov(self.viewpoint.fx * scale_factor, new_w),
                                focal2fov(self.viewpoint.fy * scale_factor, new_h),
                                new_w, new_h, delta_tau=self.viewpoint.delta_tau)

        last_render_pkg, next_render_pkg = render2(last_viewpoint, self.viewpoint, next_viewpoint,
                                                   self.gaussians, self.background)
        last_intensity_frame = self.get_intensity_frame(last_render_pkg["render"])
        next_intensity_frame = self.get_intensity_frame(next_render_pkg["render"])
        sign_delta_Ir = next_intensity_frame - last_intensity_frame

        l2_norm = torch.norm(sign_delta_Ir, p=2)
        normalized_sign_delta_Ir = sign_delta_Ir / l2_norm
        normalized_unsign_delta_Ir = torch.abs(normalized_sign_delta_Ir)

        return normalized_sign_delta_Ir, normalized_unsign_delta_Ir
