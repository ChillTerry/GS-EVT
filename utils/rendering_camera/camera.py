import numpy as np
import torch
from torch import nn
from munch import munchify

from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


class Camera(nn.Module):
    """
    A Camera module represented in the gaussian splatting rendering procedure.
    """

    def __init__(self, R, t, fovx, fovy, image_width, image_height, uid=-1, data_device="cuda"):
        """
        Initializes a new instance of the Camera module.

        Args:
            R (tensor): Rotation matrix tensor for the camera.
            t (tensor): Translation vector tensor for the camera.
            fovx (float): Horizontal field of view in degrees.
            fovy (float): Vertical field of view in degrees.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            uid (int): Unique identifier for the camera.
            data_device (str): Device string. Defaults to 'cuda' for GPU usage.

        Raises:
            Exception: If the specified device is invalid.
        """
        super(Camera, self).__init__()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Camera parameters and attributes
        self.uid = uid
        self.R = R
        self.T = t
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0  # Far clipping plane distance
        self.znear = 0.01  # Near clipping plane distance

        # For fine-tuning the camera pose
        self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=self.data_device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=self.data_device))

        # Calculate transformation matrices for the camera
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # The full projection matrix from world space directly to clip space
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # The position of the camera center in the world space
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def init_from_yaml(config):
        calib_params = munchify(config["Gaussian"]["calib_params"])
        initial_R = np.array(config["Tracking"]["initial_pose"]["rot"]["data"]).reshape(3, 3)
        initial_t = np.array(config["Tracking"]["initial_pose"]["trans"]["data"]).reshape(3,)
        fovx = focal2fov(calib_params.fx, calib_params.width)
        fovy = focal2fov(calib_params.fy, calib_params.height)
        return Camera(R=initial_R, t=initial_t, fovx=fovx, fovy=fovy,
                      image_width=calib_params.width, image_height=calib_params.height)
