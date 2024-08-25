import os
import cv2
import copy
import time
import torch
import signal
import logging
import numpy as np
from typing import List
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from utils.auxiliary import Logger
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame, EventArray
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.visualizer import get_delta_Ie_img, get_delta_Ir_img, overlay_two_imgs, save_video, save_gif


# global variable for stopping the script
stop_signal_received = False

def handle_stop_signal():
    global stop_signal_received
    stop_signal_received = True
    print("Stop signal received, preparing to save results and exit...")

signal.signal(signal.SIGINT, handle_stop_signal)  # Ctrl+C sends SIGINT
signal.signal(signal.SIGTERM, handle_stop_signal)  # kill sends SIGTERM


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
        self.gaussian_kernel_size = self.config["Event"]["gaussian_kernel_size"]
        self.intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
        self.distortion_factors = np.array(config["Event"]["distortion_factors"])
        self.converged_threshold = config["Optimizer"]["converged_threshold"]
        self.max_optim_iter = config["Optimizer"]["max_optim_iter"]
        self.save_path = config["Tracking"]["save_path"]

        self.pyramid_lvl = 3
        self.angular_vel_window = []
        self.linear_vel_window = []

        # TODO: turn the level above INFO when release the code
        self.log = Logger(name='TrackingLogger', log_file=f"{self.save_path}/tracking_log.log", level=logging.INFO)

    def check_convergence(self, losses, threshold=1e-4):
        # Consider the last 10 epochs
        if len(losses) <= 10:
            return False
        # Calculate slopes
        slopes = np.diff(losses[-11:])
        # Check if the average slope of the last few epochs is below the threshold
        average_slope = np.mean(np.abs(slopes))
        if average_slope < threshold:
            return True
        else:
            return False

    def image_pyramid(self, image):
        device = image.device
        dtype = image.dtype
        image = image.detach().cpu().numpy().transpose(1, 2, 0).squeeze(axis=-1)
        h, w = image.shape
        pyramid = []

        for level in range(self.pyramid_lvl):
            scale_factor = 0.5 ** level
            resized_img = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)),
                                     interpolation=cv2.INTER_NEAREST)
            tensor_resized_img = torch.from_numpy(np.expand_dims(resized_img, axis=0)).to(device=device, dtype=dtype)
            pyramid.append(tensor_resized_img)
        return pyramid

    def tracking_loss(self, delta_Ir, delta_Ie, mask=None, huber=False):
        if mask is not None:
            residual = delta_Ir * mask - delta_Ie
        else:
            residual = delta_Ir - delta_Ie
        if huber:   # the Huber norm doesn't appear to be effective
            loss = F.huber_loss(residual, torch.zeros_like(residual), delta=0.002, reduction='none')
            loss = torch.sum(loss)
        else:
            loss = torch.norm(residual)
        return loss

    def tracking(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(f"{self.save_path}/tracking_frames", exist_ok=True)
        tum_file = open(f"{self.save_path}/tracking_pose_tum.txt", "w")

        frame_idx = 0
        last_delta_tau = 0
        fraction_num = self.max_optim_iter / 2
        total_frame_nums = len(self.event_arrays)
        global stop_signal_received
        total_opt_start_time = time.time()
        while not stop_signal_received and frame_idx < total_frame_nums:
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
            self.viewpoint.const_vel_model((delta_tau + last_delta_tau) / 2)

            # before optimizing, backup the initial status
            curr_initial_status = copy.deepcopy([self.viewpoint.T.detach(),
                                                 self.viewpoint.R.detach(),
                                                 self.viewpoint.angular_vel.detach(),
                                                 self.viewpoint.linear_vel.detach()])

            self.log.info(f"frame_idx:\t{frame_idx} / {total_frame_nums}")
            self.log.info(f"delta_tau:\t{delta_tau:.4f}")

            eFrame = EventFrame(self.img_width, self.img_height, self.intrinsic, self.distortion_factors,
                                self.gaussian_kernel_size, self.event_arrays[frame_idx])
            sign_delta_Ie_pyramid = self.image_pyramid(eFrame.sign_delta_Ie)
            unsign_delta_Ie_pyramid = self.image_pyramid(eFrame.unsign_delta_Ie)

            for lvl in range(self.pyramid_lvl - 1, -1, -1):
                sign_delta_Ie = sign_delta_Ie_pyramid[lvl]
                unsign_delta_Ie = unsign_delta_Ie_pyramid[lvl]
                # mask = compute_mask(unsign_delta_Ie, lvl)

                # jump unsigned pose optimization in lower level for saving time
                if lvl == self.pyramid_lvl - 1:
                    opt_vel = False
                else:
                    opt_vel = True
                    start_vel_opt_iter = 0

                for param_group in optimizer.param_groups:
                    param = param_group["params"][0]
                    if param is self.viewpoint.cam_rot_delta:
                        param_group["lr"] = self.config["Optimizer"]["cam_rot_delta"]
                    if param is self.viewpoint.cam_trans_delta:
                        param_group["lr"] = self.config["Optimizer"]["cam_trans_delta"]
                    if param is self.viewpoint.cam_w_delta:
                        param_group["lr"] = self.config["Optimizer"]["cam_w_delta"]
                    if param is self.viewpoint.cam_v_delta:
                        param_group["lr"] = self.config["Optimizer"]["cam_v_delta"]

                losses = []   # for saving loss curve and computing loss slope
                optim_iter = 0
                converged = False
                opt_start_time = time.time()
                while True:
                    if not opt_vel:
                        self.viewpoint.cam_w_delta.requires_grad_(False)
                        self.viewpoint.cam_v_delta.requires_grad_(False)
                        self.viewpoint.cam_rot_delta.requires_grad_(True)
                        self.viewpoint.cam_trans_delta.requires_grad_(True)
                    else:
                        self.viewpoint.cam_w_delta.requires_grad_(True)
                        self.viewpoint.cam_v_delta.requires_grad_(True)
                        self.viewpoint.cam_rot_delta.requires_grad_(True)
                        self.viewpoint.cam_trans_delta.requires_grad_(True)

                        if 1 <= (optim_iter - start_vel_opt_iter) <= fraction_num:
                            fraction = (optim_iter - start_vel_opt_iter) / fraction_num
                        else:
                            fraction = 1

                        for param_group in optimizer.param_groups:  # adjust learning rate
                            param = param_group["params"][0]
                            if param is self.viewpoint.cam_rot_delta:
                                param_group["lr"] = self.config["Optimizer"]["cam_rot_delta"] * fraction
                            if param is self.viewpoint.cam_trans_delta:
                                param_group["lr"] = self.config["Optimizer"]["cam_trans_delta"] * fraction
                            if param is self.viewpoint.cam_w_delta:
                                param_group["lr"] = self.config["Optimizer"]["cam_w_delta"] * (1 - fraction)
                            if param is self.viewpoint.cam_v_delta:
                                param_group["lr"] = self.config["Optimizer"]["cam_v_delta"] * (1 - fraction)

                    rFrame = RenderFrame(self.viewpoint, self.gaussians, self.pipeline, self.background, lvl)
                    sign_delta_Ir = rFrame.sign_delta_Ir
                    unsign_delta_Ir = rFrame.unsign_delta_Ir

                    if not opt_vel:
                        loss = self.tracking_loss(unsign_delta_Ir, unsign_delta_Ie, huber=False)
                    else:
                        loss = self.tracking_loss(sign_delta_Ir, sign_delta_Ie, huber=False)
                    loss.backward()
                    losses.append(loss.item())

                    with torch.no_grad():
                        optimizer.step()
                        converged = self.check_convergence(losses, self.converged_threshold)
                        if not opt_vel: # coarse stage
                            self.viewpoint.update_pose()
                        else:           # fine stage
                            self.viewpoint.update_vwRT()
                        optimizer.zero_grad()

                    if converged:
                        if not opt_vel:
                            opt_vel = True
                            start_vel_opt_iter = optim_iter
                        else:
                            break

                    if not opt_vel:
                        if optim_iter >= self.max_optim_iter:
                            self.log.error("coarse stage optimization iter exceeded the max_optim_iter!")
                            break
                    else:
                        if optim_iter >= start_vel_opt_iter + self.max_optim_iter:
                            self.log.error("fine stage optimization iter exceeded the max_optim_iter!")
                            break

                    optim_iter += 1

                opt_end_time = time.time()
                opt_time = opt_end_time - opt_start_time
                self.log.info(f"level:\t{lvl}")
                self.log.info(f"optim_iter:\t{optim_iter} ({start_vel_opt_iter}+{optim_iter - start_vel_opt_iter})")
                self.log.info(f"opt_time:\t{opt_time:.4f}")

            # mix the optimized vel and vel calculated from const vel model
            if frame_idx >= 5:  # give it some warm up frames. e.g.:5
                self.viewpoint.cal_weighted_velocity(curr_initial_status[:2], (delta_tau + last_delta_tau) / 2, 0.5)
            last_delta_tau = delta_tau

            self.log.debug(f"angular_vel:\t{self.viewpoint.angular_vel}")
            self.log.debug(f"linear_vel:\t{self.viewpoint.linear_vel}")
            self.log.info("="*20)

            # find the timestamp of the event frame by taking average of start and end time
            timestamp = self.event_arrays[frame_idx].time()
            translation = self.viewpoint.T.detach().cpu().numpy()
            rotation_matrix = self.viewpoint.R.detach().cpu().numpy()
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()

            tum_file.write(f"{timestamp} {translation[0]} {translation[1]} {translation[2]} "
                           f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}\n")

            sign_delta_Ie_img = get_delta_Ie_img(eFrame.sign_delta_Ie)
            sign_delta_Ir_img = get_delta_Ir_img(rFrame.sign_delta_Ir)
            img = overlay_two_imgs(sign_delta_Ir_img, sign_delta_Ie_img)
            cv2.imwrite(os.path.join(f"{self.save_path}/tracking_frames", f'frame_{frame_idx}.png'), img)

            frame_idx += 1
            if frame_idx >= total_frame_nums:
                break
        tum_file.close()

        total_opt_end_time = time.time()
        total_opt_time = total_opt_end_time - total_opt_start_time
        self.log.info(f"totoal tracking time cost: {total_opt_time:.4f}s")

        img_dir = f"{self.save_path}/tracking_frames"
        video_save_path = f"{self.save_path}/tracking_video.mp4"
        gif_save_path = f"{self.save_path}/tracking_video.gif"
        save_video(img_dir, video_save_path, fps=120)
        save_gif(img_dir, gif_save_path, duration=2)
        self.log.info(f"save video in: {video_save_path}")
        self.log.info(f"save gif in  : {gif_save_path}")