"""
Code for loading models trained with DINOv2 SALAD 
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class DINOv2SALAD(BaseModel):
    default_conf = {
        "variant": "SALAD",
        "backbone": "DINOv2",
        # "fc_output_dim": 2048,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load("serizba/salad", "dinov2_salad").eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        with torch.no_grad():
            desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
