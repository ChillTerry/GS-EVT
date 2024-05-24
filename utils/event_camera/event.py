import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List

EVENT_BRIGHTNESS = 20  # Define the brightness increment/decrement for events.


def load_events_from_txt(data_path, max_events_per_frame, num_arrays=None):
    event_arrays = []
    # total_events = sum(1 for _ in open(data_path))
    with open(data_path, 'r', encoding='utf-8') as event_file:
        # num_frames = total_events // max_events_per_frame
        if num_arrays is not None:
            # num_frames = min(num_frames, num_arrays)
            num_frames = num_arrays
        for i in tqdm(range(num_frames), desc="load events"):
            event_array = EventArray()
            for j in range(max_events_per_frame):
                line = next(event_file)
                line_data = line.strip().split(' ')
                line_data = [int(item) for item in line_data]
                event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
                event_array.callback(event)
            event_arrays.append(event_array)
    return event_arrays


class Event:
    def __init__(self, x=None, y=None, ts=None, polarity=None):
        self.x = x
        self.y = y
        self.ts = ts
        self.polarity = polarity


class EventArray:
    def __init__(self):
        self.events: List[Event] = []

    def callback(self, event):
        """Adds new event to the event list."""
        self.events.append(event)

    def begin(self):
        return 0

    def end(self):
        return self.size()-1

    def size(self):
        return len(self.events)

    def duration(self):
        return (self.events[-1].ts - self.events[0].ts) / float(1e6)


class EventFrame:
    """An EventFrame module represented in the rendering procedure."""

    def __init__(self, img_width, img_height, intrinsic, distortion_factors, event_array: EventArray, device='cuda'):
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.intrinsic = intrinsic
        self.distortion_factors = distortion_factors
        self.delta_Ie = self.integrate_events(event_array)

    def integrate_events(self, event_array: EventArray):
        """Integrates events into the visual frame based on their coordinates and polarities."""
        event_frame = np.zeros((self.img_width, self.img_height), dtype=np.uint8)
        for i in range(event_array.begin(), event_array.end()):
            event = event_array.events[i]

            if event.x >= self.img_width or event.y >= self.img_height:
                print("WARNING: Ignoring out of bounds event at {}, {}".format(event.x, event.y))
                continue

            if event.polarity:
                event_frame[event.x, event.y] = min(event_frame[event.x, event.y] + EVENT_BRIGHTNESS, 255)
            else:
                event_frame[event.x, event.y] = max(event_frame[event.x, event.y] - EVENT_BRIGHTNESS, 0)

        event_frame = cv2.undistort(event_frame.transpose(), self.intrinsic, self.distortion_factors)
        event_frame = torch.tensor(np.expand_dims(event_frame, axis=0), device=self.device)

        min_val = event_frame.min()
        range_val = event_frame.max() - min_val
        event_frame = (event_frame - min_val) / range_val

        return event_frame
