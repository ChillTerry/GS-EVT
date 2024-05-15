import numpy as np
from typing import List

EVENT_BRIGHTNESS = 20  # Define the brightness increment/decrement for events.

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


class EventFrame:
    """An EventFrame module represented in the rendering procedure."""

    def __init__(self, img_width, img_height, event_array: EventArray):
        self.img_width = img_width
        self.img_height = img_height
        self.event_frame = self.integrate_events(event_array)

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

        return event_frame