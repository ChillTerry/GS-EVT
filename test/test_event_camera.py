import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser

from utils.event_camera.event import EventFrame
from utils.event_camera.event import load_events_from_txt


"""Render the whole sequence into an mp4 video"""
def render_video(save_dir, data_path, img_width, img_height, intrinsic, distortion_factors, max_events_per_frame, filter_threshold):
    eFrames:List[EventFrame] = []
    event_arrays = load_events_from_txt(data_path, max_events_per_frame, array_nums=None)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(save_dir, 'event_video.mp4'),
                                   fourcc, fps, (img_width, img_height))

    for i in tqdm(range(len(event_arrays)), desc="extracting event frames"):
        eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, filter_threshold, event_arrays[i])

        delta_Ie_np = eFrame.delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
        color_Ie = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
        negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
        positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
        color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
        color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)
        color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)

        video_writer.write(color_Ie)

    video_writer.release()


'''Render the first frame to see the quality'''
def render_frame(save_dir, data_path, img_width, img_height, intrinsic, distortion_factors, max_events_per_frame, filter_threshold):
    event_arrays = load_events_from_txt(data_path, max_events_per_frame, array_nums=1)
    eFrame = EventFrame(img_width, img_height, intrinsic, distortion_factors, filter_threshold, event_arrays[0])
    delta_Ie_np = eFrame.delta_Ie.detach().cpu().numpy().transpose(1, 2, 0) * 255
    color_Ie = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
    negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
    positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
    color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
    color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)
    color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, 'event_frame.png'), color_Ie)


def test_event_camera(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]
    intrinsic = np.array(config["Event"]["intrinsic"]["data"]).reshape(3, 3)
    distortion_factors = np.array(config["Event"]["distortion_factors"])
    filter_threshold = config["Event"]["filter_threshold"]

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    render_frame(results_dir, data_path, img_width, img_height, intrinsic, distortion_factors, max_events_per_frame, filter_threshold)
    # render_video(results_dir, data_path, img_width, img_height, intrinsic, distortion_factors, max_events_per_frame, filter_threshold)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/VECTOR/desk_normal1_config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_event_camera(args.config_path)
