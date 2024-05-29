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
        self.intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
        self.distortion_factors = np.array(config["Event"]["distortion_factors"])
        self.converged_threshold = config["Optimizer"]["converged_threshold"]
        self.max_optim_iter = config["Optimizer"]["max_optim_iter"]

    def tracking(self):
        overlay_imgs = []
        frame_idx = 0
        while True:
            opt_params = []
            opt_params.append({"params": [self.viewpoint.cam_rot_delta],
                            "lr": self.config["Optimizer"]["cam_rot_delta"]})

            opt_params.append({"params": [self.viewpoint.cam_trans_delta],
                            "lr": self.config["Optimizer"]["cam_trans_delta"]})

            optimizer = torch.optim.Adam(opt_params)

            delta_tau = self.event_arrays[frame_idx].duration()
            print(delta_tau)
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

                gray_Ir = delta_Ir_np.astype(np.uint8)
                gray_Ir = cv2.cvtColor(gray_Ir, cv2.COLOR_GRAY2BGR)
                if optim_iter == 0:
                    cv2.imwrite(os.path.join("./results", f'delta_Ir.png'), gray_Ir)

                color_Ie = np.zeros((delta_Ir_np.shape[0], delta_Ir_np.shape[1], 3), dtype=np.uint8)
                negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
                positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
                color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
                color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)
                color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)
                if optim_iter == 0:
                    cv2.imwrite(os.path.join("./results", f'delta_Ie.png'), color_Ie)

                # Overlay the color image onto the grayscale image using a weighted sum
                alpha = 0.5  # Define the transparency level: 0.0 - completely transparent; 1.0 - completely opaque
                overlay_img = cv2.addWeighted(color_Ie, alpha, gray_Ir, 1 - alpha, 0)
                overlay_imgs.append(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
                if optim_iter == 0:
                    cv2.imwrite(os.path.join("./results", f'tracking_frame.png'), overlay_img)
                # cv2.imshow("img", overlay_img)
                # if cv2.waitKey(0) == 27:
                #     break

                optim_iter += 1
                if converged or optim_iter >= self.max_optim_iter:
                    break
            print(f"optim_iter: {optim_iter}")

            frame_idx += 1
            if frame_idx >= len(self.event_arrays):
                break
        imageio.mimsave(os.path.join("./results", f'single_frame_tracking.gif'), overlay_imgs, 'GIF', duration=0.1)
