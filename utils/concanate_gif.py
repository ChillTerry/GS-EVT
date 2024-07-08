import os
import re
import imageio

def extract_last_frame(gif_path):
    frames = imageio.mimread(gif_path)
    last_frame = frames[-1]
    return last_frame

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    gif_folder = "/home/liutao/JYA/GS-EVT/results/gif_frames"
    output_gif_path = "./results/multi_frame_tracking.gif"

    gif_files = [f for f in os.listdir(gif_folder) if f.endswith('.gif')]
    gif_files.sort(key=natural_sort_key)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    last_frames = []
    for gif_file in gif_files:
        gif_path = os.path.join(gif_folder, gif_file)
        last_frame = extract_last_frame(gif_path)
        last_frames.append(last_frame)

    imageio.mimsave(output_gif_path, last_frames, 'GIF', duration=0.5, loop=0)
    print(f"The concatenated GIF is saved into: {output_gif_path}")

if __name__ == "__main__":
    main()
