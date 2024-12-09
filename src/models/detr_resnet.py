from .model import Model

from typing import Dict, List, Union
import torch
import torchvision
from transformers import AutoModelForImageSegmentation

import coremltools as ct


def post_process_semantic_segmentation(logits, pred_masks, target_size=None):
    # Remove the null class `[..., :-1]`
    masks_classes = logits.softmax(dim=-1)[..., :-1]
    masks_probs = pred_masks.sigmoid()

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

    if target_size is not None:
        segmentation = torch.nn.functional.interpolate(
            segmentation, size=target_size, mode="bilinear", align_corners=False
        )
    semantic_segmentation = segmentation.argmax(dim=1)
    return semantic_segmentation


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, pixel_values):
        """pixel_values are floats in the range `[0, 255]`"""
        # Apply ImageNet normalization
        n_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        n_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        pixel_values = torchvision.transforms.functional.normalize(
            pixel_values, mean=n_mean, std=n_std
        )

        outputs = self.model(pixel_values, return_dict=True)
        semantic_map = post_process_semantic_segmentation(
            outputs.logits, outputs.pred_masks, pixel_values.shape[-2:]
        )
        return semantic_map[0]


class DetrResnet(Model):
    def name():
        return "facebook/detr-resnet-50-panoptic"

    def __init__(self):
        super().__init__()

        model = AutoModelForImageSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic"
        )
        model.eval()

        self.model = Wrapper(model)

    def torch_example_input(
        self,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        return (torch.rand(1, 3, 448, 448) * 255,)

    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [ct.ImageType(name="image", shape=(1, 3, 448, 448))]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [ct.TensorType("semanticPredictions")]
