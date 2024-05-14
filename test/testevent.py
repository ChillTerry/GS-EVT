import cv2
import numpy as np
import h5py
import hdf5plugin
from EventBuffer import EventBuffer, Event, EventArray
from tqdm import tqdm
import linecache

def render_event(file, single_frame, event_buffer):
    for i in range(single_frame):
        line = next(file)
        line_data = line.strip().split(' ')
        line_data = [int(item) for item in line_data]
        event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
        ev_array = EventArray()
        ev_array.set_events([event])
        event_buffer.callback(ev_array)

    from_time = event_buffer.data[0].ts
    event_frame = event_buffer.integrate_time(from_time, single_frame)
    img = np.clip(event_frame.img, 0, 255)
    event_buffer.clear_data()
    return img

"""Render the whole sequence into an mp4 video"""
def render_mp4(event_file):
    event_buffer = EventBuffer()
    frames = []

    with open(event_file, 'r', encoding='utf-8') as file:
        for i in tqdm(range((num_lines - start_index) // single_frame)):
            frame = render_event(file, single_frame, event_buffer)
            frames.append(frame)

    print("rendering to mp4......")
    frame_height, frame_width = frames[0].shape[:2]
    fps = 30
    video_writer = cv2.VideoWriter('event_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()

'''Render the first frame to see the quality'''
def render_frame(event_file):
    event_buffer = EventBuffer()
    frames = []

    with open(event_file, 'r', encoding='utf-8') as file:
        first_frame = render_event(file, single_frame, event_buffer)
    
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1280, 720)

    cv2.imwrite('first_frame.png', first_frame) # save before show it
    cv2.imshow('Image', first_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
event_file = '/home/liutao/Dataset/GSEVT/0426_event_dataset/0426_calibration/events.txt'
num_lines = sum(1 for _ in open(event_file))
print("Total event amount: ", num_lines)

# Settings
single_frame = 100000 # how many events contribute to each frame
start_index = 0 # default to be the first event
print("Total intergration frames: ", (num_lines - start_index) // single_frame)

render_frame(event_file)
# render_mp4(event_file)

