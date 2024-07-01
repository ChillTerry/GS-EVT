import torch
from torch import nn
from munch import munchify

from utils.pose import SE3_exp
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


class Camera(nn.Module):
    """
    A Camera module represented in the gaussian splatting rendering procedure.
    """

    def __init__(self,
                 R,
                 t,
                 angular_vel,
                 linear_vel,
                 fovx,
                 fovy,
                 image_width,
                 image_height,
                 delta_tau=0,
                 device="cuda"):
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
        self.R = R.to(device)
        self.T = t.to(device)
        self.angular_vel = angular_vel
        self.linear_vel = linear_vel
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0  # Far clipping plane distance
        self.znear = 0.01  # Near clipping plane distance
        self.delta_tau = delta_tau

        # For fine-tuning the camera pose
        self.cam_rot_delta = nn.Parameter(torch.zeros(3, device=self.device), requires_grad=True)
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, device=self.device), requires_grad=True)
        self.cam_w_delta = nn.Parameter(torch.zeros(3, device=self.device), requires_grad=False)
        self.cam_v_delta = nn.Parameter(torch.zeros(3, device=self.device), requires_grad=False)

    @property
    def projection_matrix(self):
        # Calculate projection matrix for the camera
        # TODO: MonoGS uses getProjectionMatrix2, does it really make obvious difference?
        # return getProjectionMatrix2(znear=0.01, zfar=100.0, fx=346.6, fy=347.0, cx=196.5, cy=110.0,
        #                             W=self.image_width, H=self.image_height).transpose(0, 1).to(device=self.device)
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                   fovY=self.FoVy).transpose(0,1).to(device=self.device)

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

    @property
    def curr_pose(self):
        curr_pose = torch.eye(4, device=self.device)
        curr_pose[0:3, 0:3] = self.R
        curr_pose[0:3, 3] = self.T
        return curr_pose

    @property
    def last_vel_transform(self):
        rot_vec, trans_vec = self.compute_motion_vectors()
        delta_pose_vec = torch.cat([-trans_vec, -rot_vec], axis=0)
        return SE3_exp(delta_pose_vec)

    @property
    def last_vel_transform_inv(self):
        return torch.linalg.inv(self.last_vel_transform)

    @property
    def next_vel_transform(self):
        rot_vec, trans_vec = self.compute_motion_vectors()
        delta_pose_vec = torch.cat([trans_vec, rot_vec], axis=0)
        return SE3_exp(delta_pose_vec)

    @property
    def next_vel_transform_inv(self):
        return torch.linalg.inv(self.next_vel_transform)

    def compute_motion_vectors(self):
        rot_vec = self.angular_vel * (self.delta_tau / 2)
        trans_vec = self.linear_vel * (self.delta_tau / 2)
        return rot_vec, trans_vec

    def update_vwRT(self, converged_threshold):
        self.update_velocity()
        converged = self.update_pose(converged_threshold)
        return converged

    def update_pose(self, converged_threshold=5e-4):
        deltaT = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)

        T_w2c = torch.eye(4, device=self.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.T

        new_w2c = SE3_exp(deltaT) @ T_w2c
        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        converged = deltaT.norm() < converged_threshold
        self.update_RT(new_R, new_T)

        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)
        return converged

    def update_velocity(self):
        self.angular_vel += self.cam_w_delta
        self.linear_vel += self.cam_v_delta
        # print(f"w_delta:\t{self.cam_w_delta.data}")
        # print(f"v_delta:\t{self.cam_v_delta.data}")
        self.cam_w_delta.data.fill_(0)
        self.cam_v_delta.data.fill_(0)

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def const_vel_model(self, tau):
        # predict the estimated next pose according to constant velocity model
        # angular_vel is in the form of euler angle(xyz)
        rot_vec = self.angular_vel * tau
        trans_vec = self.linear_vel * tau

        delta_pose_vec1 = torch.cat([trans_vec, rot_vec], axis=0)
        delta_pose1 = SE3_exp(delta_pose_vec1)
        curr_pose = torch.eye(4, device=self.device)
        curr_pose[0:3, 0:3] = self.R
        curr_pose[0:3, 3] = self.T

        new_pose = delta_pose1 @ curr_pose
        new_R = new_pose[:3, :3]
        new_t = new_pose[:3, 3]

        self.last_R = self.R.clone()
        self.last_T = self.T.clone()
        self.update_RT(new_R, new_t)

    @staticmethod
    def init_from_yaml(config):
        img_width = config["Gaussian"]["img_width"]
        img_height = config["Gaussian"]["img_height"]
        device = config["Gaussian"]["model_params"]["device"]
        calib_params = munchify(config["Gaussian"]["calib_params"])
        R = torch.tensor(config["Tracking"]["initial_pose"]["rot"]["data"]).reshape(3, 3)
        t = torch.tensor(config["Tracking"]["initial_pose"]["trans"]["data"]).reshape(3,)
        linear_vel = torch.tensor(config["Tracking"]["initial_vel"]["linear_vel"], device=device, dtype=torch.float32)
        angular_vel = torch.tensor(config["Tracking"]["initial_vel"]["angular_vel"], device=device, dtype=torch.float32)
        fovx = focal2fov(calib_params.fx, img_width)
        fovy = focal2fov(calib_params.fy, img_height)
        viewpoint = Camera(R, t, angular_vel, linear_vel, fovx, fovy, img_width, img_height, device=device)
        viewpoint.update_pose()
        return viewpoint