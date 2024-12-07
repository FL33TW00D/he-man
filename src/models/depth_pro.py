from .model import Model

import sys
import os
from typing import Union, List, Dict
import warnings

import torch
from torchvision.transforms import Normalize
from huggingface_hub import PyTorchModelHubMixin
from depth_pro.depth_pro import (
    create_backbone_model,
    DepthPro,
    DepthProEncoder,
    MultiresConvDecoder,
)
import coremltools as ct

class DepthProWrapper(DepthPro, PyTorchModelHubMixin):
    """Depth Pro network."""

    def __init__(
        self,
        patch_encoder_preset: str,
        image_encoder_preset: str,
        decoder_features: str,
        fov_encoder_preset: str,
        use_fov_head: bool = True,
        **kwargs,
    ):
        """Initialize Depth Pro."""

        patch_encoder, patch_encoder_config = create_backbone_model(
            preset=patch_encoder_preset
        )
        image_encoder, _ = create_backbone_model(preset=image_encoder_preset)

        fov_encoder = None
        if use_fov_head and fov_encoder_preset is not None:
            fov_encoder, _ = create_backbone_model(preset=fov_encoder_preset)

        dims_encoder = patch_encoder_config.encoder_feature_dims
        hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
        encoder = DepthProEncoder(
            dims_encoder=dims_encoder,
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
            hook_block_ids=hook_block_ids,
            decoder_features=decoder_features,
        )
        decoder = MultiresConvDecoder(
            dims_encoder=[encoder.dims_encoder[0]] + list(encoder.dims_encoder),
            dim_decoder=decoder_features,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            last_dims=(32, 1),
            use_fov_head=use_fov_head,
            fov_encoder=fov_encoder,
        )


class DepthProInvDepthNormalized(DepthProWrapper):
    @torch.inference_mode
    def forward(self, x):
        x_norm = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)

        canonical_inverse_depth, _ = super().forward(x_norm)

        min_vals = canonical_inverse_depth.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = canonical_inverse_depth.amax(dim=(1, 2, 3), keepdim=True)
        inverse_depth_normalized = (canonical_inverse_depth - min_vals) / (
            max_vals - min_vals
        )

        return inverse_depth_normalized * 255.0

class DepthPro(Model):
    def name():
        return "apple/DepthPro-mixin"
    
    def __init__(self):
        super().__init__()

        depthpro_pytorch_inv_depth_norm = DepthProInvDepthNormalized.from_pretrained(
            DepthPro.name()
        )
        self.model = depthpro_pytorch_inv_depth_norm.eval()

        self.cached_torch_trace = None
        self.cached_coreml_model = None

    def torch_module(self) -> torch.nn.Module:
        if self.cached_torch_trace:
            return self.cached_torch_trace
        
        # There should be a function that does this in a with block, god only knows why I must suffer
        # I belive it has something to do with this:
        # https://stackoverflow.com/questions/75022490/pytorch-torch-no-grad-doesnt-affect-modules
        original_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            traced_model = torch.jit.trace(self.model, (self.torch_example_input(),))
            return traced_model
        
        torch.set_grad_enabled(original_grad_state)

    def torch_example_input(self) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        return torch.rand((1, 3, 1536, 1536))
    
    def coreml_model(self) -> ct.models.MLModel:
        if self.cached_coreml_model:
            return self.cached_coreml_model

        ct_model = ct.convert(
            self.torch_module(),
            inputs=[
                ct.ImageType(
                    name="image",
                    color_layout=ct.colorlayout.RGB,
                    shape=(1, 3, 1536, 1536),
                    scale=1 / 255.0,
                ),
            ],
            outputs=[
                ct.ImageType(
                    name="normalizedInverseDepth",
                    color_layout=ct.colorlayout.GRAYSCALE_FLOAT16,
                )
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )

        self.cached_coreml_model = ct_model

        return ct_model