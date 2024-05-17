import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import yaml
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser
from utils.event_camera.event import EventFrame, EventArray, Event


def extract_single_event_frame(event_file, img_width, img_height, max_events_per_frame):
    event_array = EventArray()
    for i in range(max_events_per_frame):
        line = next(event_file)
        line_data = line.strip().split(' ')
        line_data = [int(item) for item in line_data]
        event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
        event_array.callback(event)
    return EventFrame(img_width, img_height, event_array)


"""Render the whole sequence into an mp4 video"""
def render_video(save_path, data_path, img_width, img_height, max_events_per_frame):
    eFrames:List[EventFrame] = []
    total_events = sum(1 for _ in open(data_path))
    with open(data_path, 'r', encoding='utf-8') as event_file:
        for i in tqdm(range(total_events // max_events_per_frame), desc="extracting event frames"):
            eFrame = extract_single_event_frame(event_file, img_width, img_height, max_events_per_frame)
            eFrames.append(eFrame)
            # cv2.imwrite(os.path.join(save_path, f'event_frame_{i}.png'), eFrame.event_frame.transpose())

    print("saving to mp4 video...")
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(save_path, 'event_video.mp4'),
                                   fourcc, fps, (img_width, img_height))

    for eFrame in eFrames:
        frame = cv2.cvtColor(eFrame.event_frame.transpose(), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()
    print("done!")


'''Render the first frame to see the quality'''
def render_frame(data_path, img_width, img_height, max_events_per_frame):
    with open(data_path, 'r', encoding='utf-8') as event_file:
        eFrame = extract_single_event_frame(event_file, img_width, img_height, max_events_per_frame)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', eFrame.img_width, eFrame.img_height)
    cv2.imshow('Image', eFrame.event_frame.transpose())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_event_camera(config_path):
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)
    data_path = config["Event"]["data_path"]
    max_events_per_frame = config["Event"]["max_events_per_frame"]
    img_width = config["Event"]["img_width"]
    img_height = config["Event"]["img_height"]

    results_path = os.path.join(BASE_DIR, "results")
    os.makedirs(results_path, exist_ok=True)
    results_path = os.path.join(BASE_DIR, "results")

    # render_frame(data_path, img_width, img_height, max_events_per_frame)
    render_video(results_path, data_path, img_width, img_height, max_events_per_frame)


if __name__ == "__main__":
    parser = ArgumentParser(description="configuration parameters")
    parser.add_argument("--config_path", "-c", type=str, default="./configs/config.yaml")
    args = parser.parse_args(sys.argv[1:])

    test_event_camera(args.config_path)
