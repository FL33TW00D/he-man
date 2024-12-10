from abc import abstractmethod
from typing import Dict, List, Union, Any

import torch
import coremltools as ct
from PIL import Image
import numpy as np
import warnings


class Model:
    def __init__(self):
        self.cached_torch_trace = None
        self.cached_coreml_model = None

    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def recommended_iterations(self) -> int:
        pass

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
            traced_model = torch.jit.trace(self.model, self.torch_example_input())

        torch.set_grad_enabled(original_grad_state)

        self.cached_torch_trace = traced_model
        return traced_model

    @abstractmethod
    def torch_example_input(self) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def coreml_model(self) -> ct.models.MLModel:
        if self.cached_coreml_model:
            return self.cached_coreml_model

        ct_model = ct.convert(
            self.torch_module(),
            inputs=self.coreml_inputs(),
            outputs=self.coreml_outputs(),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )

        self.cached_coreml_model = ct_model

        return ct_model

    @abstractmethod
    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        pass

    @abstractmethod
    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        pass

    def coreml_example_input(self) -> Dict[str, Any]:
        inputs = {}

        ct_model = self.coreml_model()

        try:
            spec = ct_model.get_spec()
        except AttributeError:
            spec = ct_model.spec if hasattr(ct_model, "spec") else ct_model

        for input_desc in spec.description.input:
            input_name = input_desc.name

            if hasattr(input_desc, "type"):
                if input_desc.type.HasField("imageType"):
                    image_type = input_desc.type.imageType
                    height = image_type.height
                    width = image_type.width

                    if image_type.colorSpace == 0:  # GRAYSCALE
                        channels = 1
                    else:  # Default to 3
                        channels = 3

                    noise_array = np.random.randint(
                        0, 256, (height, width, channels), dtype=np.uint8
                    )
                    inputs[input_name] = Image.fromarray(noise_array)
                elif input_desc.type.HasField("multiArrayType"):
                    shape = tuple(input_desc.type.multiArrayType.shape)
                    inputs[input_name] = np.random.randn(*shape)
                else:
                    raise Exception(f"Could not determine input type for {input_name}")
            else:
                raise Exception(f"Could not determine input type for {input_name}")

        return inputs
