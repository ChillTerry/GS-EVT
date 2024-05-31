import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List

EVENT_BRIGHTNESS = 20  # Define the brightness increment/decrement for events.


def load_events_from_txt(data_path, max_events_per_frame, array_nums=None):
    event_arrays = []

    with open(data_path, 'r', encoding='utf-8') as event_file:
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
        return len(self.events)

    def duration(self):
        if self.size() > 0:
            start_ts = self.events[0].ts
            end_ts = self.events[-1].ts
            return (end_ts - start_ts) / 1e6
        return 0


class EventFrame:
    """An EventFrame module represented in the rendering procedure."""

    def __init__(self, img_width, img_height, intrinsic, distortion_factors,
                 filter_threshold, event_array: EventArray, device='cuda'):
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.intrinsic = intrinsic
        self.distortion_factors = distortion_factors
        self.filter_threshold = filter_threshold
        self.delta_Ie = self.integrate_events(event_array)

    def integrate_events(self, event_array: EventArray):
        """Integrates events into the visual frame based on their coordinates and polarities."""
        delta_Ie = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        for event in event_array.events:
            delta_Ie[event.y, event.x] += 1 if event.polarity else -1
        delta_Ie =cv2.undistort(delta_Ie, self.intrinsic, self.distortion_factors)
        delta_Ie = cv2.GaussianBlur(delta_Ie, (5, 5), 0)
        max_val = delta_Ie.max()
        min_val = delta_Ie.min()
        abs_max_val = max(max_val, np.abs(min_val))
        delta_Ie = delta_Ie / abs_max_val
        # delta_Ie = ((delta_Ie - min_val) / (max_val - min_val))
        if abs_max_val == max_val:
            pos_filter_threshold = self.filter_threshold
            neg_filter_threshold = self.filter_threshold * (np.abs(min_val) / abs_max_val)
        else:
            pos_filter_threshold = self.filter_threshold * (max_val / abs_max_val)
            neg_filter_threshold = self.filter_threshold
        print(f"pos_filter_threshold: {pos_filter_threshold}")
        print(f"neg_filter_threshold: {neg_filter_threshold}")
        delta_Ie[(delta_Ie > -neg_filter_threshold) & (delta_Ie < pos_filter_threshold)] = 0

        # # Normalize values in the range [min_val, 0] to [-1, 0]
        # negative_part_mask = delta_Ie < 0
        # delta_Ie[negative_part_mask] = delta_Ie[negative_part_mask] / abs(min_val)

        # # Normalize values in the range [0, max_val] to [0, 1]
        # positive_part_mask = delta_Ie >= 0
        # delta_Ie[positive_part_mask] = delta_Ie[positive_part_mask] / max_val

        # import matplotlib.pyplot as plt
        # plt.close()
        # plt.imshow(delta_Ie, cmap='viridis', interpolation='none')
        # plt.colorbar(label='Value')
        # plt.title('2D Array Distribution')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.savefig('delta_Ie_filter_abs_below_0.png', dpi=300, bbox_inches='tight')

        delta_Ie = torch.tensor(np.expand_dims(delta_Ie, axis=0), device=self.device)
        return delta_Ie
