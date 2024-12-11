from .model import Model

import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Union, List, Dict
from PIL import Image
import numpy as np
import coremltools as ct


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )

        next_tokens = torch.argmax(outputs[0], dim=-1)
        return next_tokens


class BlipCaption(Model):
    def name():
        return "Salesforce/blip-image-captioning-base"

    def recommended_iterations(self) -> int:
        return 100
    
    def __init__(self):
        super().__init__()

        self.processor = BlipProcessor.from_pretrained(
            BlipCaption.name()
        )
        self.raw_model = BlipForConditionalGeneration.from_pretrained(
            BlipCaption.name()
        )
        self.model = Wrapper(self.raw_model).eval()

    def torch_example_input(
        self,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        img_url = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
        )
        raw_image = Image.open(requests.get(img_url, stream=True).raw)

        pixel_values = raw_image.convert("RGB").resize((384, 384), Image.BICUBIC)
        pixel_values = np.array(pixel_values) * (1 / 255)
        CLIP_MEAN = np.array([0.48145467, 0.4578275, 0.40821072])
        CLIP_STD = np.array([0.26862955, 0.2613026, 0.2757771])
        pixel_values = (pixel_values - CLIP_MEAN) / CLIP_STD
        pixel_values = pixel_values.transpose((2, 0, 1))
        pixel_values = torch.tensor(np.array([pixel_values]), dtype=torch.float32)

        input_ids = torch.LongTensor([[0] * 100])
        input_ids = input_ids.to(dtype=torch.int32)
        input_ids[0, 0] = self.raw_model.config.text_config.bos_token_id

        attention_mask = torch.LongTensor([[0] * 100])
        attention_mask = attention_mask.to(dtype=torch.int32)
        attention_mask[0, 0] = 1

        return (pixel_values, input_ids, attention_mask)

    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [
            ct.TensorType(
                name="pixel_values", shape=(1, 3, 384, 384), dtype=np.float32
            ),
            ct.TensorType(name="input_ids", shape=(1, 100), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 100), dtype=np.int32),
        ]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [ct.TensorType(name="output_ids", dtype=np.int32)]
