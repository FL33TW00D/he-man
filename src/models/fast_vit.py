import numpy as np
from .model import Model

import torch
import timm
import warnings
from typing import Union, List, Dict
from urllib.request import urlopen
from PIL import Image
import coremltools as ct


class FastVit(Model):
    def name():
        return "timm/fastvit_ma36.apple_in1k"

    def __init__(self):
        super().__init__()

        self.model = timm.create_model("fastvit_ma36.apple_in1k", pretrained=True)
        self.model = self.model.eval()

    def torch_example_input(
        self,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        img = Image.open(urlopen(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ))

        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        return (transforms(img).unsqueeze(0),)
    
    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [
            ct.ImageType(name="image", shape=(1,3,256,256))
        ]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [
            ct.TensorType(
                name="output_ids",
                dtype=np.int32
            )
        ]