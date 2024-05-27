import os
import cv2
import torch
import imageio
import numpy as np
from typing import List

from utils.pose import update_pose
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame, EventArray
from gaussian_splatting.scene.gaussian_model import GaussianModel


def tracking_loss(delta_Ir, delta_Ie):
    return torch.abs((delta_Ir - delta_Ie)).mean()


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
        self.intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
        self.distortion_factors = np.array(config["Event"]["distortion_factors"])
        self.converged_threshold = config["Optimizer"]["converged_threshold"]
        self.max_optim_iter = config["Optimizer"]["max_optim_iter"]

    def tracking(self):
        merge_imgs = []
        frame_idx = 0
        while True:
            opt_params = []
            opt_params.append({"params": [self.viewpoint.cam_rot_delta],
                            "lr": self.config["Optimizer"]["cam_rot_delta"]})

            opt_params.append({"params": [self.viewpoint.cam_trans_delta],
                            "lr": self.config["Optimizer"]["cam_trans_delta"]})

            optimizer = torch.optim.Adam(opt_params)

            delta_tau = self.event_arrays[frame_idx].duration()
            # print(delta_tau)
            eFrame = EventFrame(self.img_width, self.img_height, self.intrinsic,
                                self.distortion_factors, self.event_arrays[frame_idx])
            delta_Ie = eFrame.delta_Ie

            optim_iter = 0
            while True:
                rFrame = RenderFrame(self.viewpoint, self.gaussians, self.pipeline, self.background)
                delta_Ir = rFrame.get_delta_Ir(delta_tau)

                loss = tracking_loss(delta_Ir, delta_Ie)
                loss.backward()
                # print(f"loss: {loss.item()}")
                # print(f"delta_rot grad  : {self.viewpoint.cam_rot_delta.grad}")
                # print(f"delta_trans grad: {self.viewpoint.cam_trans_delta.grad}")
                with torch.no_grad():
                    optimizer.step()
                    converged = update_pose(self.viewpoint, self.converged_threshold)
                    optimizer.zero_grad()

                delta_Ie_np = delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
                delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0) * 255
                merge_img = np.zeros((delta_Ir_np.shape[0], delta_Ir_np.shape[1], 3), dtype=np.uint8)
                merge_img[:, :, 0] = delta_Ir_np.squeeze(axis=-1)
                merge_img[:, :, 2] = delta_Ie_np.squeeze(axis=-1)
                merge_imgs.append(merge_img)

                # if frame_idx == 0:
                #     merge_img = np.zeros((delta_Ir_np.shape[0], delta_Ir_np.shape[1], 3), dtype=np.uint8)
                #     merge_img[:, :, 0] = delta_Ir_np.squeeze(axis=-1)
                #     frame = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(os.path.join("./results", f'delta_Ir.png'), frame)

                #     merge_img = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
                #     merge_img[:, :, 2] = delta_Ie_np.squeeze(axis=-1)
                #     frame = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(os.path.join("./results", f'delta_Ie.png'), frame)

                if converged or optim_iter >= self.max_optim_iter:
                    break
                # break
                optim_iter += 1
            print(f"optim_iter: {optim_iter}")
            frame_idx += 1
            break
        imageio.mimsave(os.path.join("./results", f'single_frame_tracking.gif'), merge_imgs, 'GIF', duration=0.1)
