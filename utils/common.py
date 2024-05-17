import os
import torch
from tqdm import tqdm

from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import Event, EventArray

def load_events_from_txt(data_path, max_events_per_frame):
    event_arrays = []
    total_events = sum(1 for _ in open(data_path))
    with open(data_path, 'r', encoding='utf-8') as event_file:
        for i in tqdm(range(total_events // max_events_per_frame), desc="load events"):
            event_array = EventArray()
            for i in range(max_events_per_frame):
                line = next(event_file)
                line_data = line.strip().split(' ')
                line_data = [int(item) for item in line_data]
                event = Event(line_data[1], line_data[2], line_data[0], line_data[3])
                event_array.callback(event)
            event_arrays.append(event_array)
    return event_arrays


def save_render_image(rFrame: RenderFrame, id=None):
    import torchvision
    from torchvision.transforms.functional import to_pil_image
    if id is not None:
        depth_image_name = f"depth_{id}.png"
        color_image_name = f"color_{id}.png"
    else:
        depth_image_name = f"depth.png"
        color_image_name = f"color.png"

    results_path = "./results"
    os.makedirs(results_path, exist_ok=True)
    render_image = rFrame.color_frame
    render_depth = rFrame.depth_frame
    # Save images
    min_val = torch.min(render_depth)
    max_val = torch.max(render_depth)
    normalized_depth_tensor = (render_depth - min_val) / (max_val - min_val)
    normalized_depth_tensor = torch.clamp(normalized_depth_tensor, 0, 1)
    depth_image = to_pil_image(normalized_depth_tensor)
    depth_image.save(os.path.join(results_path, depth_image_name))

    torchvision.utils.save_image(render_image, os.path.join(results_path, color_image_name))


def tracking_loss(event_frame, render_frame):
    return torch.abs((render_frame - event_frame)).mean()
