import os
import cv2
import numpy as np
from PIL import Image
from natsort import natsorted

def get_delta_Ie_img(delta_Ie):
    """Generates and returns an image based on self.sign_delta_Ie."""
    if isinstance(delta_Ie, np.ndarray):
        delta_Ie_np = np.expand_dims(delta_Ie, axis=2)
    else:
        delta_Ie_np = delta_Ie.detach().cpu().numpy().transpose(1, 2, 0)

    original_max = delta_Ie_np.max()
    original_min = delta_Ie_np.min()
    abs_max_val = max(original_max, abs(original_min))
    delta_Ie_np = delta_Ie_np * (255 / abs_max_val)

    color_Ie = np.zeros((delta_Ie_np.shape[0], delta_Ie_np.shape[1], 3), dtype=np.uint8)
    negative_delta_Ie_np = np.where(delta_Ie_np < 0, delta_Ie_np, 0)
    positive_delta_Ie_np = np.where(delta_Ie_np > 0, delta_Ie_np, 0)
    color_Ie[:, :, 0] = positive_delta_Ie_np.squeeze(axis=-1)
    color_Ie[:, :, 2] = -negative_delta_Ie_np.squeeze(axis=-1)
    color_Ie = cv2.cvtColor(color_Ie, cv2.COLOR_RGB2BGR)
    return color_Ie


def get_delta_Ir_img(delta_Ir):
    delta_Ir_np = delta_Ir.detach().cpu().numpy().transpose(1, 2, 0)

    original_max = delta_Ir_np.max()
    original_min = delta_Ir_np.min()
    abs_max_val = max(original_max, abs(original_min))
    delta_Ir_np = delta_Ir_np * (255 / abs_max_val)
    gray_Ir = (delta_Ir_np + 255) / 2
    gray_Ir = gray_Ir.astype(np.uint8)
    gray_Ir = cv2.cvtColor(gray_Ir, cv2.COLOR_GRAY2BGR)
    return gray_Ir


def overlay_two_imgs(img1, img2):
    # Overlay the color image onto the grayscale image using a weighted sum
    alpha = 0.6  # Define the transparency level: 0.0 - completely transparent; 1.0 - completely opaque
    img = cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)
    return img


def save_video(img_dir, save_path, fps=120):
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    image_files = natsorted(image_files)
    image_paths = [os.path.join(img_dir, f) for f in image_files]

    image = cv2.imread(image_paths[0])
    height, width, layers = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 120, (width, height))

    for image_path in image_paths:
        video.write(cv2.imread(image_path))
    video.release()


def save_gif(img_dir, save_path, duration=10):
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    image_files = natsorted(image_files)
    image_paths = [os.path.join(img_dir, f) for f in image_files]
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    images[0].save(save_path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)   # duration is in ms