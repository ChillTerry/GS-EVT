import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
import torch
import numpy as np
from typing import List
from munch import munchify
from argparse import ArgumentParser

from utils.pose import update_pose
from utils.tracker import tracking_loss
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.event_camera.event import EventFrame, EventArray, load_events_from_txt


def test_single_frame_tracking(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    model_params = munchify(config["Gaussian"]["model_params"])
    pipeline = munchify(config["Gaussian"]["pipeline_params"])
    device = model_params.device
    background = torch.tensor(model_params.background, dtype=torch.float32, device=device)
    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]
    intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
    distortion_factors = np.array(config["Event"]["distortion_factors"])
    linear_vel = torch.tensor(config["Tracking"]["initial_vel"]["linear_vel"], device=device, dtype=torch.float32)
    angular_vel = torch.tensor(config["Tracking"]["initial_vel"]["angular_vel"], device=device, dtype=torch.float32)


    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    viewpoint = Camera.init_from_yaml(config)
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.load_ply(model_params.model_path)

    event_arrays: List[EventArray] = load_events_from_txt(data_path, max_events_per_frame, array_nums=1)
    delta_tau = event_arrays[0].duration()

    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, event_arrays[0])
    delta_Ie = eFrame.delta_Ie

    delta_tau = 0.172009
    viewpoint.T += 0.3   # Add disturbance on translation
    rFrame = RenderFrame(viewpoint, gaussians, pipeline, background)
    delta_Ir = rFrame.get_delta_Ir(delta_tau, linear_vel, angular_vel)

    delta_Ie_np = delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
    delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0) * 255
    merge_img = np.zeros((delta_Ir_np.shape[0], delta_Ir_np.shape[1], 3), dtype=np.uint8)
    merge_img[:, :, 0] = delta_Ir_np.squeeze(axis=-1)
    merge_img[:, :, 2] = delta_Ie_np.squeeze(axis=-1)
    frame = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(results_dir, f'tracking_0.png'), frame)

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(results_dir, 'single_frame_tracking.mp4'),
                                   fourcc, fps, (img_width, img_height))

    opt_params = []
    opt_params.append({"params": [viewpoint.cam_rot_delta],
                        "lr": config["Optimizer"]["cam_rot_delta"]})

    opt_params.append({"params": [viewpoint.cam_trans_delta],
                        "lr": config["Optimizer"]["cam_trans_delta"]})

    optimizer = torch.optim.Adam(opt_params)
    cnt = 0
    while True:
        cnt += 1
        optimizer.zero_grad()
        loss = tracking_loss(delta_Ir, delta_Ie)
        loss.backward()
        print(f"loss: {loss.item()}")
        print(f"delta_rot grad  : {viewpoint.cam_rot_delta.grad}")
        print(f"delta_trans grad: {viewpoint.cam_trans_delta.grad}")
        with torch.no_grad():
            optimizer.step()
            converged = update_pose(viewpoint)

        delta_Ie_np = delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
        delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0) * 255
        merge_img = np.zeros((delta_Ir_np.shape[0], delta_Ir_np.shape[1], 3), dtype=np.uint8)
        merge_img[:, :, 0] = delta_Ir_np.squeeze(axis=-1)
        merge_img[:, :, 2] = delta_Ie_np.squeeze(axis=-1)
        frame = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        cv2.imwrite(os.path.join(results_dir, f'tracking_{cnt}.png'), frame)

        # if converged or cnt >= 10:
        #     break
        if cnt >= 100:
            break
        print(f"iter: {cnt}")

        delta_Ir = rFrame.get_delta_Ir(delta_tau, linear_vel, angular_vel)

    video_writer.release()


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_single_frame_tracking(args.config_path)
