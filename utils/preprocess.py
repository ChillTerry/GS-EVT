import os
import shutil
import glob
import numpy as np

def select_images(dataset_path, gaussian_splatting_data_path, num_images=150):
		for seq_dir in os.listdir(dataset_path):
				seq_path = os.path.join(dataset_path, seq_dir)
				if not os.path.isdir(seq_path):
						continue
				
				gs_seq_path = os.path.join(gaussian_splatting_data_path, seq_dir)
				os.makedirs(gs_seq_path, exist_ok=True)
				
				input_path = os.path.join(gs_seq_path, 'input')
				os.makedirs(input_path, exist_ok=True)
				
				rgb_files = sorted(glob.glob(os.path.join(seq_path, 'rgb', '*.png')))
				
				interval = max(len(rgb_files) // num_images, 1)
				
				selected_images = []
				timestamps = []

				count = 0
				for i in range(0, len(rgb_files), interval):
						image_path = rgb_files[i]
						image_name = os.path.basename(image_path)
						selected_images.append(image_name)
						shutil.copy(image_path, os.path.join(input_path, f'frame_{count}.png'))
						timestamps.append(image_name)
						count += 1
				
				with open(os.path.join(seq_path, 'timestamps.txt'), 'w') as f:
						f.write('\n'.join(timestamps))
				
				print(f"Processed {seq_dir}: Selected {len(selected_images)} images.")

dataset_path = '/home/liutao/JYA/BIG_BOARD_DATASET'
gaussian_splatting_data_path = '/home/liutao/JYA/gaussian-splatting/data/BIG_BOARD'

##------------------------------------------------------pick 150 images from original sequence -------------------------------------------------------------
# select_images(dataset_path, gaussian_splatting_data_path)

gaussian_splatting_path = '/home/liutao/JYA/gaussian-splatting'

import subprocess

##-------------------------------------------------------------do colmap for all sequence-------------------------------------------------------------------
# folders = os.listdir(gaussian_splatting_data_path)
# for folder in folders:
# 		convert_cmd = f"python {os.path.join(gaussian_splatting_path, 'convert.py')} -s {os.path.join(gaussian_splatting_data_path, folder)}"
# 		print(f"Running command: {convert_cmd}")
# 		subprocess.run(convert_cmd, shell=True, check=True)
		
# config_command = "unset LD_LIBRARY_PATH"
# subprocess.run(config_command, shell=True, check=True)
##-------------------------------------------------------train gaussian splatting for all sequence----------------------------------------------------------
# for folder in folders:
# 		train_cmd = f"python {os.path.join(gaussian_splatting_path, 'train.py')} -s {os.path.join(gaussian_splatting_data_path, folder)} -m {os.path.join(gaussian_splatting_path, 'output', folder)}"
# 		print(f"Running command: {train_cmd}")
# 		subprocess.run(train_cmd, shell=True, check=True)

# print("All folders processed.")

##-----------------------------------------------------------generate config yamls & data--------------------------------------------------------------------
import yaml
import json

config_folder_path = "/home/liutao/JYA/GS-EVT/configs/BIG_BOARD"
GSEVT_data_path = "/home/liutao/JYA/GS-EVT/data/BIG_BOARD"
template_path = '/home/liutao/JYA/GS-EVT/configs/config_template.yaml'

# # Load template config
# with open(template_path, 'r') as f:
# 	template_config = yaml.safe_load(f)

folders = os.listdir(gaussian_splatting_data_path)
# for folder in folders:
# 	# Step 1: Create a new folder for each sequence under GSEVT_data_path
# 	seq_folder_path = os.path.join(GSEVT_data_path, folder)
# 	os.makedirs(seq_folder_path, exist_ok=True)
	
# 	# Step 2: Copy 'input' folder from gaussian_splatting_data_path to seq_folder_path
# 	input_src = os.path.join(gaussian_splatting_data_path, folder, 'input')
# 	input_dest = os.path.join(seq_folder_path, 'input')
# 	shutil.copytree(input_src, input_dest)
	
# 	print(f"Copied 'input' folder from {gaussian_splatting_data_path}/{folder} to {seq_folder_path}")

# 	# Additional Step: Copy 'distorted' folder from gaussian_splatting_data_path to seq_folder_path
# 	distorted_src = os.path.join(gaussian_splatting_data_path, folder, 'distorted')
# 	distorted_dest = os.path.join(seq_folder_path, 'distorted')
# 	shutil.copytree(distorted_src, distorted_dest)
	
# 	# Step 3: Copy point_cloud.ply to seq_folder_path
# 	point_cloud_src = os.path.join(gaussian_splatting_path, 'output', folder, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
# 	point_cloud_dest = os.path.join(seq_folder_path, 'point_cloud.ply')
# 	shutil.copy(point_cloud_src, point_cloud_dest)
# 	print(f"Copied point_cloud.ply to {seq_folder_path}")

# 	# Step 4: Read cameras.json and extract fx, fy
# 	cameras_json_path = os.path.join(gaussian_splatting_path, 'output', folder, 'cameras.json')
# 	with open(cameras_json_path, 'r') as f:
# 		cameras_data = json.load(f)
	
# 	fx = cameras_data[0]['fx']
# 	fy = cameras_data[0]['fy']
	
# 	# Step 5: Create config file path
# 	config_file_path = os.path.join(config_folder_path, f'{folder}_config.yaml')

# 	# Step 6: Make a copy of the template config
# 	config = template_config.copy()
	
# 	# Step 7: Update Gaussian:calib_params with fx and fy
# 	config['Gaussian']['calib_params']['fx'] = fx
# 	config['Gaussian']['calib_params']['fy'] = fy
	
# 	# Step 8: Update Gaussian:model_params:model_path with point_cloud.ply path
# 	config['Gaussian']['model_params']['model_path'] = os.path.join(seq_folder_path, 'point_cloud.ply')
	
# 	# Step 9: Update Event:data_path with events.txt path
# 	config['Event']['data_path'] = os.path.join(seq_folder_path, "events.txt")

# 	# Step 10: Save updated config to file
# 	with open(config_file_path, 'w') as f:
# 		yaml.dump(config, f)

# 	print(f"Created config file for {folder}: {config_file_path}")

# print("Task completed: Copied input folders and point_cloud.ply for each sequence.")

##-----------------------------------------------------------calculate initial velocity--------------------------------------------------------------------

import os
import collections
import numpy as np
import struct
import argparse
from scipy.spatial.transform import Rotation as R
 
 
CameraModel = collections.namedtuple(
	"CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
	"Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
	"Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
	"Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
 
 
class Image(BaseImage):
	def qvec2rotmat(self):
		return qvec2rotmat(self.qvec)
 
 
CAMERA_MODELS = {
	CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
	CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
	CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
	CameraModel(model_id=3, model_name="RADIAL", num_params=5),
	CameraModel(model_id=4, model_name="OPENCV", num_params=8),
	CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
	CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
	CameraModel(model_id=7, model_name="FOV", num_params=5),
	CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
	CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
	CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
						 for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
						   for camera_model in CAMERA_MODELS])
 
 
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
	"""Read and unpack the next bytes from a binary file.
	:param fid:
	:param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
	:param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
	:param endian_character: Any of {@, =, <, >, !}
	:return: Tuple of read and unpacked values.
	"""
	data = fid.read(num_bytes)
	return struct.unpack(endian_character + format_char_sequence, data)
 
def read_cameras_text(path):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::WriteCamerasText(const std::string& path)
		void Reconstruction::ReadCamerasText(const std::string& path)
	"""
	cameras = {}
	with open(path, "r") as fid:
		while True:
			line = fid.readline()
			if not line:
				break
			line = line.strip()
			if len(line) > 0 and line[0] != "#":
				elems = line.split()
				camera_id = int(elems[0])
				model = elems[1]
				width = int(elems[2])
				height = int(elems[3])
				params = np.array(tuple(map(float, elems[4:])))
				cameras[camera_id] = Camera(id=camera_id, model=model,
											width=width, height=height,
											params=params)
	return cameras
 
def read_cameras_binary(path_to_model_file):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::WriteCamerasBinary(const std::string& path)
		void Reconstruction::ReadCamerasBinary(const std::string& path)
	"""
	cameras = {}
	with open(path_to_model_file, "rb") as fid:
		num_cameras = read_next_bytes(fid, 8, "Q")[0]
		for _ in range(num_cameras):
			camera_properties = read_next_bytes(
				fid, num_bytes=24, format_char_sequence="iiQQ")
			camera_id = camera_properties[0]
			model_id = camera_properties[1]
			model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
			width = camera_properties[2]
			height = camera_properties[3]
			num_params = CAMERA_MODEL_IDS[model_id].num_params
			params = read_next_bytes(fid, num_bytes=8*num_params,
									 format_char_sequence="d"*num_params)
			cameras[camera_id] = Camera(id=camera_id,
										model=model_name,
										width=width,
										height=height,
										params=np.array(params))
		assert len(cameras) == num_cameras
	return cameras
 
def read_images_text(path):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::ReadImagesText(const std::string& path)
		void Reconstruction::WriteImagesText(const std::string& path)
	"""
	images = {}
	with open(path, "r") as fid:
		while True:
			line = fid.readline()
			if not line:
				break
			line = line.strip()
			if len(line) > 0 and line[0] != "#":
				elems = line.split()
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				camera_id = int(elems[8])
				image_name = elems[9]
				elems = fid.readline().split()
				xys = np.column_stack([tuple(map(float, elems[0::3])),
									   tuple(map(float, elems[1::3]))])
				point3D_ids = np.array(tuple(map(int, elems[2::3])))
				images[image_id] = Image(
					id=image_id, qvec=qvec, tvec=tvec,
					camera_id=camera_id, name=image_name,
					xys=xys, point3D_ids=point3D_ids)
	return images
 
 
def read_images_binary(path_to_model_file):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::ReadImagesBinary(const std::string& path)
		void Reconstruction::WriteImagesBinary(const std::string& path)
	"""
	images = {}
	with open(path_to_model_file, "rb") as fid:
		num_reg_images = read_next_bytes(fid, 8, "Q")[0]
		for _ in range(num_reg_images):
			binary_image_properties = read_next_bytes(
				fid, num_bytes=64, format_char_sequence="idddddddi")
			image_id = binary_image_properties[0]
			qvec = np.array(binary_image_properties[1:5])
			tvec = np.array(binary_image_properties[5:8])
			camera_id = binary_image_properties[8]
			image_name = ""
			current_char = read_next_bytes(fid, 1, "c")[0]
			while current_char != b"\x00":   # look for the ASCII 0 entry
				image_name += current_char.decode("utf-8")
				current_char = read_next_bytes(fid, 1, "c")[0]
			num_points2D = read_next_bytes(fid, num_bytes=8,
										   format_char_sequence="Q")[0]
			x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
									   format_char_sequence="ddq"*num_points2D)
			xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
								   tuple(map(float, x_y_id_s[1::3]))])
			point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
			images[image_id] = Image(
				id=image_id, qvec=qvec, tvec=tvec,
				camera_id=camera_id, name=image_name,
				xys=xys, point3D_ids=point3D_ids)
 
	return images 

def read_points3D_text(path):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::ReadPoints3DText(const std::string& path)
		void Reconstruction::WritePoints3DText(const std::string& path)
	"""
	points3D = {}
	with open(path, "r") as fid:
		while True:
			line = fid.readline()
			if not line:
				break
			line = line.strip()
			if len(line) > 0 and line[0] != "#":
				elems = line.split()
				point3D_id = int(elems[0])
				xyz = np.array(tuple(map(float, elems[1:4])))
				rgb = np.array(tuple(map(int, elems[4:7])))
				error = float(elems[7])
				image_ids = np.array(tuple(map(int, elems[8::2])))
				point2D_idxs = np.array(tuple(map(int, elems[9::2])))
				points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
											   error=error, image_ids=image_ids,
											   point2D_idxs=point2D_idxs)
	return points3D
 
def read_points3D_binary(path_to_model_file):
	"""
	see: src/base/reconstruction.cc
		void Reconstruction::ReadPoints3DBinary(const std::string& path)
		void Reconstruction::WritePoints3DBinary(const std::string& path)
	"""
	points3D = {}
	with open(path_to_model_file, "rb") as fid:
		num_points = read_next_bytes(fid, 8, "Q")[0]
		for _ in range(num_points):
			binary_point_line_properties = read_next_bytes(
				fid, num_bytes=43, format_char_sequence="QdddBBBd")
			point3D_id = binary_point_line_properties[0]
			xyz = np.array(binary_point_line_properties[1:4])
			rgb = np.array(binary_point_line_properties[4:7])
			error = np.array(binary_point_line_properties[7])
			track_length = read_next_bytes(
				fid, num_bytes=8, format_char_sequence="Q")[0]
			track_elems = read_next_bytes(
				fid, num_bytes=8*track_length,
				format_char_sequence="ii"*track_length)
			image_ids = np.array(tuple(map(int, track_elems[0::2])))
			point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
			points3D[point3D_id] = Point3D(
				id=point3D_id, xyz=xyz, rgb=rgb,
				error=error, image_ids=image_ids,
				point2D_idxs=point2D_idxs)
	return points3D

 
def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)) and \
       os.path.isfile(os.path.join(path, "points3D" + ext)):
        print("Detected model format: '" + ext + "'")
        return True
 
    return False
 
def read_model(path, ext=""):
	# try to detect the extension automatically
	if ext == "":
		if detect_model_format(path, ".bin"):
			ext = ".bin"
		elif detect_model_format(path, ".txt"):
			ext = ".txt"
		else:
			print("Provide model format: '.bin' or '.txt'")
			return
 
	if ext == ".txt":
		cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
		images = read_images_text(os.path.join(path, "images" + ext))
		points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
	else:
		cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
		images = read_images_binary(os.path.join(path, "images" + ext))
		points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
	return cameras, images, points3D
 
 
def write_model(cameras, images, points3D, path, ext=".bin"):
	if ext == ".txt":
		write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
		write_images_text(images, os.path.join(path, "images" + ext))
		write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
	else:
		write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
		write_images_binary(images, os.path.join(path, "images" + ext))
		write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
	return cameras, images, points3D
 
 
def qvec2rotmat(qvec):
	return np.array([
		[1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
		 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
		 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
		[2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
		 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
		 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
		[2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
		 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
		 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
 
 
def rotmat2qvec(R):
	Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
	K = np.array([
		[Rxx - Ryy - Rzz, 0, 0, 0],
		[Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
		[Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
		[Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
	eigvals, eigvecs = np.linalg.eigh(K)
	qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
	if qvec[0] < 0:
		qvec *= -1
	return qvec

for folder in folders:
	seq_folder_path = os.path.join(GSEVT_data_path, folder)
	# path to colmap output
	colmap_path = os.path.join(seq_folder_path, 'distorted', 'sparse', '0')
	cameras, images, points3D = read_model(path=colmap_path)

	print("num_cameras:", len(cameras))
	print("num_images:", len(images))
	print("num_points3D:", len(points3D))

	first = list(images.keys())[0]
	second = list(images.keys())[1]

	try:
		# first image pose(world to regular camera)
		first_image = images[first]
		qvec = first_image.qvec
		tvec = first_image.tvec
		rotmat = qvec2rotmat(qvec)
		T_rw = np.eye(4)  
		T_rw[:3, :3] = rotmat  
		T_rw[:3, 3] = tvec  

		# T_re = [ 0.9999407352369797  , 0.009183655542749752,  0.005846920950435052,  0.0005085820608404798, 
		# 				-0.009131364645448854, 0.9999186289230431  , -0.008908070070089353, -0.04081979450823404  ,
		# 				-0.005928253827254812, 0.008854151768176144,  0.9999432282899994  , -0.0140781304960408   ,
		# 				0                   , 0                   ,  0                   ,  1                    ]    # transformation from left event camera to left regular camera(vector stuff)
		# T_re = np.array(T_re).reshape((4, 4))
		# T_er = np.linalg.inv(T_re)

		# T_ew = T_er @ T_rw
		# R_ew = T_ew[:3, :3]
		# t_ew = T_ew[:3, 3]

		# find initial velocity(calculated with frame1 and frame 5)
		fifth_image = images[second]
		qvec5 = fifth_image.qvec
		tvec5 = fifth_image.tvec
		rotmat5 = qvec2rotmat(qvec5)
		T_5w = np.eye(4)
		T_5w[:3, :3] = rotmat5
		T_5w[:3, 3] = tvec5
		T_w1 = np.linalg.inv(T_rw)

		T_51 = T_5w @ T_w1
		R_51 = T_51[:3, :3]
		t_51 = T_51[:3, 3]

		rotation_51 = R.from_matrix(R_51)
		axis_angle_51 = rotation_51.as_rotvec()

		timestamp_file = os.path.join(dataset_path, folder, 'timestamps.txt')

		with open(timestamp_file, 'r') as file:
			lines = file.readlines()

		first_timestamp_str = lines[first - 1].strip()
		fifth_timestamp_str = lines[second - 1].strip()

		first_timestamp = int(first_timestamp_str.split('.')[0])
		fifth_timestamp = int(fifth_timestamp_str.split('.')[0])

		time_difference = (fifth_timestamp - first_timestamp) / 1e6

		initial_angular_velocity = axis_angle_51 / time_difference
		initial_linear_velocity = t_51 / time_difference

		config_file_path = os.path.join(config_folder_path, f'{folder}_config.yaml')
		with open(config_file_path, 'r') as f:
			template_config = yaml.safe_load(f)

		config = template_config.copy()

		template_config['Tracking']['initial_pose']['rot']['data'] = rotmat.flatten().tolist()
		template_config['Tracking']['initial_pose']['trans']['data'] = tvec.flatten().tolist()

		template_config['Tracking']['initial_vel']['angular_vel'] = initial_angular_velocity.tolist()
		template_config['Tracking']['initial_vel']['linear_vel'] = initial_linear_velocity.tolist()

		print("rot 数据:", template_config['Tracking']['initial_pose']['rot']['data'])
		print("trans 数据:", template_config['Tracking']['initial_pose']['trans']['data'])
		print("angular_vel:", template_config['Tracking']['initial_vel']['angular_vel'])
		print("linear_vel:", template_config['Tracking']['initial_vel']['linear_vel'])


		with open(config_file_path, 'w') as f:
			yaml.dump(config, f)
		
		print(f"writing to {config_file_path}")
	except KeyError:
		print("捕获到 KeyError 错误！")
		print(f"发生在{folder}")

import os

for folder in folders:
    try:
        print(f"processing {folder}")
        seq_folder_path = os.path.join(GSEVT_data_path, folder)
        timestamp_file = os.path.join(dataset_path, folder, 'timestamps.txt')

        # 读取timestamps.txt文件中的第一个时间戳
        with open(timestamp_file, 'r') as file:
            first_timestamp_str = file.readline().strip()
        
        first_timestamp = int(first_timestamp_str.split('.')[0])

        # 处理events.txt文件，删除早于first_timestamp的行
        events_file = os.path.join(seq_folder_path, "events.txt")
        temp_file = os.path.join(seq_folder_path, "events_temp.txt")

        with open(events_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                parts = line.split()
                event_timestamp = parts[0][:16]  # 取前16位数字
                if int(event_timestamp) >= int(first_timestamp):
                    outfile.write(f"{event_timestamp} {' '.join(parts[1:])}\n")

        # 将临时文件重命名为原始文件名
        os.remove(events_file)
        os.rename(temp_file, events_file)
    except FileNotFoundError:
        print(f"File '{timestamp_file}' not found.")