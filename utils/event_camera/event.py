import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List

EVENT_BRIGHTNESS = 20  # Define the brightness increment/decrement for events.


def load_events_from_txt(data_path, max_events_per_frame, array_nums=None):
    event_arrays = []

    with open(data_path, 'r', encoding='utf-8') as event_file:
        # Preallocate event_array only if we're not limiting the number of arrays
        preallocate_size = max_events_per_frame

        event_array = EventArray()
        event_count = 0

        for line in tqdm(event_file, desc="load events"):
            line_data = [int(item) for item in line.strip().split(' ')]
            event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
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
        return len(self.events)  # Count non-None events

    def duration(self):
        if self.size() > 0:
            start_ts = self.events[0].ts
            end_ts = self.events[-1].ts
            return (end_ts - start_ts) / 1e6
        return 0


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
        for i in range(event_array.size()):
            event = event_array.events[i]
            event_frame[event.x, event.y] = event_frame[event.x, event.y] + EVENT_BRIGHTNESS

            # if event.x >= self.img_width or event.y >= self.img_height:
            #     print("WARNING: Ignoring out of bounds event at {}, {}".format(event.x, event.y))
            #     continue

            # if event.polarity:
            #     event_frame[event.x, event.y] = min(event_frame[event.x, event.y] + EVENT_BRIGHTNESS, 255)
            # else:
            #     event_frame[event.x, event.y] = max(event_frame[event.x, event.y] - EVENT_BRIGHTNESS, 0)

        # print(f"event_frame max: {event_frame.max()}")
        event_frame = np.clip(event_frame, 0, 255)
        event_frame = cv2.undistort(event_frame.transpose(), self.intrinsic, self.distortion_factors)
        event_frame = cv2.GaussianBlur(event_frame, ksize=(5, 5), sigmaX=0)
        event_frame = torch.tensor(np.expand_dims(event_frame, axis=0), device=self.device)

        min_val = 0
        range_val = event_frame.max() - min_val
        event_frame = (event_frame - min_val) / range_val

        return event_frame
