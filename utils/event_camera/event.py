import cv2
import torch
import numpy as np
from typing import List

EVENT_BRIGHTNESS = 1  # Define the brightness increment/decrement for events.


def load_events_from_txt(data_path, max_events_per_frame, array_nums=None, start_time=None):
    event_arrays = []

    with open(data_path, 'r', encoding='utf-8') as event_file:
        event_array = EventArray()
        event_count = 0
        
        for line in event_file:
            line_data = [int(item) for item in line.strip().split(' ')]
            event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
            if start_time is not None and np.int(event.ts) < start_time:
                continue
            event_array.callback(event)
            event_count += 1

            if event_count == max_events_per_frame:
                event_arrays.append(event_array)
                event_array = EventArray()
                event_count = 0

                if array_nums is not None and len(event_arrays) >= array_nums:
                    break

        # if event_count:  # In case there are leftover events
        #     event_arrays.append(event_array)

    return event_arrays


class Event:
    __slots__ = ['x', 'y', 'ts', 'polarity']  # Using __slots__ for performance

    def __init__(self, x=None, y=None, ts=None, polarity=None):
        self.x = x
        self.y = y
        self.ts = ts
        self.polarity = polarity


class EventArray:
    def __init__(self):
        self.events: List[Event] = []

    def callback(self, event):
        self.events.append(event)

    def size(self):
        return len(self.events)

    def duration(self):
        if self.size() > 0:
            start_ts = self.events[0].ts
            end_ts = self.events[-1].ts
            return (end_ts - start_ts) / 1e6
        return 0

    def time(self):
        return (self.events[0].ts + (self.events[-1].ts - self.events[0].ts) / 2) / 1e6


class EventFrame:
    """An EventFrame module represented in the rendering procedure."""

    def __init__(self, img_width, img_height, intrinsic, distortion_factors,
                 gaussian_kernel_size, event_array: EventArray, device='cuda'):
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.intrinsic = intrinsic
        self.distortion_factors = distortion_factors
        self.gaussian_kernel_size = gaussian_kernel_size
        self.sign_delta_Ie, self.unsign_delta_Ie = self.integrate_events(event_array)

    def integrate_events(self, event_array: EventArray):
        """Integrates events into the visual frame based on their coordinates and polarities."""
        sign_delta_Ie = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        for event in event_array.events:
            sign_delta_Ie[event.y, event.x] += EVENT_BRIGHTNESS if event.polarity else -EVENT_BRIGHTNESS
        sign_delta_Ie = cv2.undistort(sign_delta_Ie, self.intrinsic, self.distortion_factors)
        sign_delta_Ie = cv2.GaussianBlur(sign_delta_Ie, (self.gaussian_kernel_size, self.gaussian_kernel_size),
                                         0, borderType=cv2.BORDER_REPLICATE)
        sign_delta_Ie = cv2.normalize(sign_delta_Ie, None)
        sign_delta_Ie = torch.tensor(np.expand_dims(sign_delta_Ie, axis=0), device=self.device)
        unsign_delta_Ie = torch.abs(sign_delta_Ie)

        return sign_delta_Ie, unsign_delta_Ie
