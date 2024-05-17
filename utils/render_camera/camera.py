import numpy as np
import torch
from torch import nn
from munch import munchify

from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


class Camera(nn.Module):
    """
    A Camera module represented in the gaussian splatting rendering procedure.
    """

    def __init__(self, R, t, fovx, fovy, image_width, image_height, uid=-1, device="cuda"):
        """
        Initializes a new instance of the Camera module.

        Args:
            R (tensor): Rotation matrix tensor for the camera in world coordinate.
            t (tensor): Translation vector tensor for the camera in world coordinate.
            fovx (float): Horizontal field of view in degrees.
            fovy (float): Vertical field of view in degrees.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            uid (int): Unique identifier for the camera.
            device (str): Device string. Defaults to 'cuda' for GPU usage.

        Raises:
            Exception: If the specified device is invalid.
        """
        super(Camera, self).__init__()
        try:
            self.device = torch.device(device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {device} failed, fallback to default cuda device" )
            self.device = torch.device("cuda")

        # Camera parameters and attributes
        self.uid = uid
        self.R = torch.tensor(R, device=device)
        self.T = torch.tensor(t, device=device)
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0  # Far clipping plane distance
        self.znear = 0.01  # Near clipping plane distance

        # For fine-tuning the camera pose
        self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=self.device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=self.device))

        # Calculate projection matrix for the camera
        # TODO: MonoGS uses getProjectionMatrix2, does it really make obvious difference?
        # self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=346.6, fy=347.0, cx=196.5, cy=110.0,
        #                                               W=self.image_width, H=self.image_height).transpose(0, 1).to(device=self.device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device=self.device)

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    @staticmethod
    def init_from_yaml(config):
        img_width = config["Event"]["img_width"]
        img_height = config["Event"]["img_height"]
        calib_params = munchify(config["Gaussian"]["calib_params"])
        initial_R = np.array(config["Tracking"]["initial_pose"]["rot"]["data"]).reshape(3, 3)
        initial_t = np.array(config["Tracking"]["initial_pose"]["trans"]["data"]).reshape(3,)
        fovx = focal2fov(calib_params.fx, img_height)
        fovy = focal2fov(calib_params.fy, img_height)
        return Camera(R=initial_R, t=initial_t, fovx=fovx, fovy=fovy,
                      image_width=img_width, image_height=img_height)
