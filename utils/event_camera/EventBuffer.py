import numpy as np
import cv2
from itertools import tee

event_brightness = 20

class Eventframe:
    def __init__(self):
        self.from_event = None
        self.to_event = None
        self.img = None


class Event:
    def __init__(self, x = None, y = None, ts = None, polarity = None):
        self.x = x
        self.y = y
        self.ts = ts
        self.polarity = polarity


class EventArray:
    def __init__(self):
        self.Header = None
        self.height = None
        self.width = None
        self.events = None
    
    def set_events(self, event_array):
        self.events = event_array


class EventBuffer:
    """A EventBuffer module represented in the rendering procedure.
    """
    def __init__(self, camera=None):
        self.data = []
        self.camera = camera

    def set_camera(self, camera):
        self.camera = camera

    def callback(self, ev_array):
        # feed in events here
        self.data.extend(ev_array.events)

    def find(self, t):
        index, _ = next(((index, event) for index, event in enumerate(self.data) if event.ts >= t), (None, None))
        return index

    def get(self):
        return self.data

    def begin(self):
        return 0

    def end(self):
        return self.size()-1

    def size(self):
        return len(self.data)

    def clear_data(self):
        self.data = []

    def integrate_index(self, from_index, to_index):
        ef = Eventframe()
        ef.from_event = from_index
        ef.to_event = to_index
        # ef.img = np.zeros((480, 640), dtype=np.uint8) # grey scale image matrix
        ef.img = np.zeros((260, 346), dtype=np.uint8)
        # Integrate
        for i in range(from_index, to_index):
            it = self.data[i]
            if it.x >= ef.img.shape[1]:
                raise AssertionError("WARNING: Ignoring out of bounds event at {}, {}".format(it.x, it.y))
            if it.y >= ef.img.shape[0]:
                raise AssertionError("WARNING: Ignoring out of bounds event at {}, {}".format(it.x, it.y))
            if it.x >= ef.img.shape[1] or it.y >= ef.img.shape[0]:
                print("WARNING: Ignoring out of bounds event at {}, {}".format(it.x, it.y))
            else:
                if it.polarity:
                    if (ef.img[it.y, it.x] + event_brightness) > 255:
                        ef.img[it.y, it.x] = 255
                    else:   
                        ef.img[it.y, it.x] += event_brightness
                else: 
                    if (ef.img[it.y, it.x] - event_brightness) < 0:
                        ef.img[it.y, it.x] = 0
                    else:
                        ef.img[it.y, it.x] -= event_brightness
        return ef

    def integrate_time(self, from_time, num, from_middle_t=False):
        if self.size() == 0:
            raise ValueError("ERROR: cannot integrate events: Buffer is empty!")
        from_index = self.find(from_time)
        if from_index == self.end():
            print("ERROR: invalid integration window requested: start time is {}, but our data goes from {} to {}".format(from_time, data[0].ts, data[-1].ts))
            raise Exception("start time not yet available")
        if from_middle_t:
            if from_index < num / 2:
                raise ValueError("start time not available")
            from_index -= num // 2
        to_index = from_index
        if self.end() - to_index + 1 < num:
            raise ValueError("end time not yet available")
        to_index += num
        return self.integrate_index(from_index, to_index)

    def plot(self, dst, from_index, to_index):
        if dst is None or dst.size == 0:
            raise ValueError("Assertion failed: dst is empty or None")
        if dst.shape[2] == 1: # transform it to a BGR image if channel number is 1
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        scale_x = dst.shape[1] / 640
        scale_y = dst.shape[0] / 480
        if dst.dtype == np.uint8:
            if scale_x > 1 and scale_y > 1:
                for i in range(from_index, to_index):
                    it = self.data[i]
                    x0 = int(it.x * scale_x)
                    y0 = int(it.y * scale_y)
                    for y in range(int(scale_y)):
                        for x in range(int(scale_x)):
                            dst[y0+y, x0+x] = [255, 0, 0] if it.polarity else [0, 0, 255]
            else:
                for i in range(from_index, to_index):
                    it = self.data[i]
                    dst[int(it.y * scale_y), int(it.x * scale_x)] = [255, 0, 0] if it.polarity else [0, 0, 255]
        else:
            for i in range(from_index, to_index):
                it = self.data[i]
                dst[int(it.y * scale_y), int(it.x * scale_x)] = [1, 0, 0] if it.polarity else [0, 0, 1]