import torch
import torch.nn.functional as F

from utils.pose import SE3_exp
from utils.render_camera.camera import Camera
from gaussian_splatting.gaussian_renderer import render, render2
from gaussian_splatting.scene.gaussian_model import GaussianModel


def get_intensity_frame(color_frame):
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    weights = weights.to(color_frame.device)
    intensity_frame = (color_frame * weights).sum(dim=1)
    return intensity_frame


class RenderFrame:
    """
    Frame that rendered by gaussian splatting
    """

    def __init__(self, viewpoint: Camera, gaussians: GaussianModel, pipeline, background):
        self.viewpoint = viewpoint
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background

    def render(self):
        render_pkg = render(self.viewpoint, self.gaussians, self.pipeline, self.background)
        return render_pkg

    @property
    def intensity_frame(self):
        rendering = render(self.viewpoint, self.gaussians, self.pipeline, self.background)
        color_frame = rendering["render"]
        return get_intensity_frame(color_frame)

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

    def get_delta_Ir(self):
        assert self.viewpoint.delta_tau != 0, f"delta_tau should not be zero, \
        it should be same as the time interval of event frame"

        last_viewpoint_pose = self.viewpoint.last_vel_transform @ self.viewpoint.curr_pose
        next_viewpoint_pose = self.viewpoint.next_vel_transform @ self.viewpoint.curr_pose
        last_viewpoint_R = last_viewpoint_pose[:3, :3]
        last_viewpoint_t = last_viewpoint_pose[:3, 3]
        next_viewpoint_R = next_viewpoint_pose[:3, :3]
        next_viewpoint_t = next_viewpoint_pose[:3, 3]

        last_viewpoint = Camera(last_viewpoint_R, last_viewpoint_t, self.viewpoint.angular_vel,
                                self.viewpoint.linear_vel, self.viewpoint.FoVx, self.viewpoint.FoVy,
                                self.viewpoint.image_width, self.viewpoint.image_height,
                                delta_tau=self.viewpoint.delta_tau)
        next_viewpoint = Camera(next_viewpoint_R, next_viewpoint_t, self.viewpoint.angular_vel,
                                self.viewpoint.linear_vel, self.viewpoint.FoVx, self.viewpoint.FoVy,
                                self.viewpoint.image_width, self.viewpoint.image_height,
                                delta_tau=self.viewpoint.delta_tau)

        last_render_pkg, next_render_pkg = render2(last_viewpoint, self.viewpoint, next_viewpoint,
                                                   self.gaussians, self.background)
        last_intensity_frame = get_intensity_frame(last_render_pkg["render"])
        next_intensity_frame = get_intensity_frame(next_render_pkg["render"])
        delta_Ir = next_intensity_frame - last_intensity_frame

        max_val = delta_Ir.max()
        min_val = delta_Ir.min()
        abs_max_val = max(max_val, torch.abs(min_val))
        delta_Ir = delta_Ir / abs_max_val
        # delta_Ir = ((delta_Ir - min_val) / (max_val - min_val))
        # delta_Ir[(delta_Ir > -0.1) & (delta_Ir < 0.1)] = 0

        # import matplotlib.pyplot as plt
        # delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0)
        # plt.close()
        # plt.imshow(delta_Ir_np, cmap='viridis', interpolation='none')
        # plt.colorbar(label='Value')
        # plt.title('2D Array Distribution')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.savefig('delta_Ir_filter_abs_below_0.png', dpi=300, bbox_inches='tight')

        return delta_Ir
