import cv2
import torch
import numpy as np
import pandas as pd
import time

EVENT_BRIGHTNESS = 1  # Define the brightness increment/decrement for events.

def load_events_from_txt(data_path, start_time=None):
    print(f"Start loading events from {data_path}")
    start_time = time.time()
    # event_data = np.loadtxt(data_path, delimiter=' ')
    df = pd.read_csv(data_path, sep=' ', header=None, 
                     names=['ts', 'x', 'y', 'polarity'], encoding='utf-8')
    # type: (ts, x, y, polarity) shape: (N, 4)
    event_data = np.array(df.to_records(index=False), 
                          dtype=[('ts', 'int64'), ('x', 'int32'), 
                                 ('y', 'int32'), ('polarity', 'int32')])
    print(f"Loading events takes {(time.time() - start_time):.2f} seconds")
    return event_data


class EventFrame:
    """An EventFrame module represented in the rendering procedure."""

    def __init__(self, img_width, img_height, intrinsic, distortion_factors,
                 camera_model, gaussian_kernel_size, events: np.ndarray, device='cuda'):
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.intrinsic = intrinsic
        self.distortion_factors = distortion_factors
        self.gaussian_kernel_size = gaussian_kernel_size
        self.events = events
        self.camera_model = camera_model

    def integrate_events(self):
        """Integrates events into the visual frame based on their coordinates and polarities."""
        sign_delta_Ie = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        for event in self.events:
            sign_delta_Ie[event['y'], event['x']] += EVENT_BRIGHTNESS if event['polarity'] else -EVENT_BRIGHTNESS

        if self.camera_model == "pinehole":
            sign_delta_Ie = cv2.undistort(sign_delta_Ie, self.intrinsic, self.distortion_factors)
        elif self.camera_model == "fisheye":
            sign_delta_Ie = cv2.fisheye.undistortImage(sign_delta_Ie, self.intrinsic, self.distortion_factors, Knew=self.intrinsic)
        else:
            raise Exception(f"Invalid camera model: {self.camera_model}")
        
        sign_delta_Ie = cv2.GaussianBlur(sign_delta_Ie, (self.gaussian_kernel_size, self.gaussian_kernel_size),
                                         0, borderType=cv2.BORDER_REPLICATE)
        sign_delta_Ie = cv2.normalize(sign_delta_Ie, None)
        sign_delta_Ie = torch.tensor(np.expand_dims(sign_delta_Ie, axis=0), device=self.device)
        unsign_delta_Ie = torch.abs(sign_delta_Ie)

        return sign_delta_Ie, unsign_delta_Ie

    def duration(self):
        if len(self.events) > 0:
            start_ts = self.events[0]['ts']
            end_ts = self.events[-1]['ts']
            return (end_ts - start_ts) / 1e6
        return 0
    
    def time(self):
        return (self.events[0]['ts'] + (self.events[-1]['ts'] - self.events[0]['ts']) / 2) / 1e6
