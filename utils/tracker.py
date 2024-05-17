import torch
from typing import List

from utils.common import tracking_loss
from utils.render_camera.camera import Camera
from utils.render_camera.frame import RenderFrame
from utils.event_camera.event import EventFrame, EventArray, Event
from gaussian_splatting.scene.gaussian_model import GaussianModel


class Tracker:
    def __init__(self,
                 config,
                 event_array: List[Event],
                 viewpoint: Camera,
                 gaussians: GaussianModel,
                 pipeline,
                 background):
        self.config = config
        self.event_array = event_array
        self.viewpoint = viewpoint
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.img_width = self.config["Event"]["img_width"]
        self.img_height = self.config["Event"]["img_height"]

    def tracking(self):
        opt_params = []
        opt_params.append({"params": [self.viewpoint.cam_rot_delta],
                           "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                           "name": "rot_{}".format(self.viewpoint.uid)})

        opt_params.append({"params": [self.viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(self.viewpoint.uid)})

        optimizer = torch.optim.Adam(opt_params)

        frame_idx = 0
        while True:
            eFrame = EventFrame(self.img_width, self.img_height, self.event_array[frame_idx])
            rFrame = RenderFrame(self.viewpoint, self.gaussians, self.pipeline, self.background)
            loss = tracking_loss(eFrame.event_frame, rFrame.intensity_frame)
            loss.backward()
