import os
import cv2
import time
import torch
import imageio
import numpy as np
from typing import List

from utils.pose import update_pose
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame, EventArray
from gaussian_splatting.scene.gaussian_model import GaussianModel


def overlay_img(delta_Ir, delta_Ie, id=None):
    delta_Ie_np = delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
    delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0) * 255

    gray_Ir = (delta_Ir_np + 255) / 2
    gray_Ir = gray_Ir.astype(np.uint8)
    gray_Ir = cv2.cvtColor(gray_Ir, cv2.COLOR_GRAY2BGR)
    if id is not None:
        cv2.imwrite(os.path.join("./results", f'delta_Ir_{id}.png'), gray_Ir)

    color_Ie = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
    negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
    positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
    color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
    color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)
    color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)
    if id is not None:
        cv2.imwrite(os.path.join("./results", f'delta_Ie_{id}.png'), color_Ie)

    # Overlay the color image onto the grayscale image using a weighted sum
    alpha = 0.5  # Define the transparency level: 0.0 - completely transparent; 1.0 - completely opaque
    img = cv2.addWeighted(color_Ie, alpha, gray_Ir, 1 - alpha, 0)
    if id is not None:
        cv2.imwrite(os.path.join("./results", f'overlay_img_{id}.png'), img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tracking_loss(delta_Ir, delta_Ie):
    return torch.norm((delta_Ir - delta_Ie))


class Tracker:
    def __init__(self,
                 config,
                 event_arrays: List[EventArray],
                 viewpoint: Camera,
                 gaussians: GaussianModel,
                 pipeline,
                 background,
                 device):
        self.device = device
        self.config = config
        self.event_arrays = event_arrays
        self.viewpoint = viewpoint
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.img_width = self.config["Event"]["img_width"]
        self.img_height = self.config["Event"]["img_height"]
        self.filter_threshold = self.config["Event"]["filter_threshold"]
        self.intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
        self.distortion_factors = np.array(config["Event"]["distortion_factors"])
        self.converged_threshold = config["Optimizer"]["converged_threshold"]
        self.max_optim_iter = config["Optimizer"]["max_optim_iter"]

    def tracking(self):
        last_imgs = []
        frame_idx = 0
        last_delta_tau = 0
        os.makedirs("./results/gif_frames", exist_ok=True)

        while True:
            overlay_imgs = []
            opt_params = []
            opt_params.append({"params": [self.viewpoint.cam_rot_delta],
                               "lr": self.config["Optimizer"]["cam_rot_delta"]})

            opt_params.append({"params": [self.viewpoint.cam_trans_delta],
                               "lr": self.config["Optimizer"]["cam_trans_delta"]})

            opt_params.append({"params": [self.viewpoint.cam_w_delta],
                                "lr": self.config["Optimizer"]["cam_w_delta"]})

            opt_params.append({"params": [self.viewpoint.cam_v_delta],
                               "lr": self.config["Optimizer"]["cam_v_delta"]})

            optimizer = torch.optim.Adam(opt_params)

            delta_tau = self.event_arrays[frame_idx].duration()
            self.viewpoint.delta_tau = delta_tau

            eFrame = EventFrame(self.img_width, self.img_height, self.intrinsic, self.distortion_factors,
                                self.filter_threshold, self.event_arrays[frame_idx])
            delta_Ie = eFrame.delta_Ie

            self.viewpoint.const_vel_model((delta_tau + last_delta_tau) / 2)

            optim_iter = 0
            opt_start_time = time.time()
            while True:
                rFrame = RenderFrame(self.viewpoint, self.gaussians, self.pipeline, self.background)
                delta_Ir = rFrame.get_delta_Ir()

                loss = tracking_loss(delta_Ir, delta_Ie)
                loss.backward()

                with torch.no_grad():
                    optimizer.step()
                    converged = update_pose(self.viewpoint, self.converged_threshold)
                    optimizer.zero_grad()

                if frame_idx == 0 and optim_iter == 0:
                    img = overlay_img(delta_Ir, delta_Ie, frame_idx)
                else:
                    img = overlay_img(delta_Ir, delta_Ie)
                overlay_imgs.append(img)

                optim_iter += 1
                if converged or optim_iter >= self.max_optim_iter:
                    break
            opt_end_time = time.time()
            opt_time = opt_end_time - opt_start_time
            print(f"frame_idx:\t{frame_idx}")
            print(f"optim_iter:\t{optim_iter}")
            print(f"opt_time:\t{opt_time:.4f}")
            print(f"delta_tau:\t{delta_tau:.4f}")
            print("="*20)

            if frame_idx != 0:
                self.viewpoint.update_velocity((delta_tau + last_delta_tau) / 2)
            # print(f"angular_vel: {self.viewpoint.angular_vel}")
            # print(f"linear_vel: {self.viewpoint.linear_vel}")

            last_delta_tau = delta_tau
            last_imgs.append(img)

            imageio.mimsave(os.path.join("./results/gif_frames", f'tracking_frame{frame_idx}.gif'),
                            overlay_imgs, 'GIF', duration=0.1)

            frame_idx += 1
            if frame_idx >= len(self.event_arrays):
                break

        imageio.mimsave(os.path.join("./results", f'multi_frame_tracking.gif'),
                        last_imgs, 'GIF', duration=0.5)