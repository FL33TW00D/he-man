from abc import abstractmethod
from typing import Dict, List, Optional, Union, Any

import torch
import PIL
import coremltools as ct
import numpy as np
import warnings


class Model:
    def __init__(self):
        self.cached_torch_trace = None
        self.cached_coreml_input = None
        self.coreml_model = None
        self.model = None

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def recommended_iterations(self) -> int:
        pass

    def torch_module(self) -> torch.nn.Module:
        if self.cached_torch_trace:
            return self.cached_torch_trace

        original_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            traced_model = torch.jit.trace(self.model, self.torch_example_input())

        torch.set_grad_enabled(original_grad_state)

        self.cached_torch_trace = traced_model
        return traced_model

    def clean_torch_cache(self):
        self.cached_torch_trace = None

    @abstractmethod
    def torch_example_input(self) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def generate_coreml_model(self) -> ct.models.MLModel:
        if self.coreml_model:
            return self.coreml_model

        cml_model = ct.convert(
            self.torch_module(),
            inputs=self.coreml_conversion_inputs(),
            outputs=self.coreml_outputs(),
            states=self.coreml_states(),
            minimum_deployment_target=ct.target.iOS18,
        )

        self.coreml_model = cml_model
        return cml_model

    """
    Define the input types for the CoreML model conversion.

    Example:
        [
            ct.TensorType(shape=(1, 2), dtype=np.int32, name="input_ids"),
            ct.TensorType(shape=(1, 1, 2, 5), dtype=np.float16, name="attention_mask"),
        ]
    """

    @abstractmethod
    def coreml_conversion_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        pass

    """ 
    Define the state types for the CoreML model conversion.

    Useful for KV cache in transformers.

    Requires iOS 18 or later.
    Example:
        [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(32, 1, 8, 128), dtype=np.float16),
                name="keyCache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(32, 1, 8, 128), dtype=np.float16),
                name="valueCache",
            ),
        ]
    """

    @abstractmethod
    def coreml_states(self) -> Optional[List[ct.StateType]]:
        pass

    """
    Define the output types for the CoreML model conversion.

    Example:
        [ct.TensorType(dtype=np.float16, name="logits")]
    """

    @abstractmethod
    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        pass

    def _generate_PIL_image(self, input_desc: ct.TensorType) -> PIL.Image:
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
        return PIL.Image.fromarray(noise_array)

    """ 
    When profiling a CoreML model, we need to provide a sample input to the
    model. This function should return a dictionary of input names to input
    values.

    Can be overriden for autoregressive models.
    """
    @abstractmethod
    def coreml_sample_input(self) -> Dict[str, Any]:
        if self.cached_coreml_input:
            return self.cached_coreml_input

        inputs = {}

        ct_model = self.generate_coreml_model()

        try:
            spec = ct_model.get_spec()
        except AttributeError:
            spec = ct_model.spec if hasattr(ct_model, "spec") else ct_model

        for input_desc in spec.description.input:
            input_name = input_desc.name

            if hasattr(input_desc, "type"):
                if input_desc.type.HasField("imageType"):
                    inputs[input_name] = self._generate_PIL_image(input_desc)
                elif input_desc.type.HasField("multiArrayType"):
                    shape = tuple(input_desc.type.multiArrayType.shape)
                    inputs[input_name] = np.random.randn(*shape)
                else:
                    raise Exception(f"Could not determine input type for {input_name}")
            else:
                raise Exception(f"Could not determine input type for {input_name}")

        self.cached_coreml_input = inputs

        return inputs

    """
    Run the model on the given compute units for the given number of iterations.
    """
    @abstractmethod
    def coreml_profile(self, compute_units: ct.ComputeUnit, model_iterations: int):
        ct_model = self.generate_coreml_model()
        mlmodel = ct.models.CompiledMLModel(ct_model.get_compiled_model_path(), compute_unit=compute_units)

        sample_input = self.coreml_sample_input()
        for _ in range(model_iterations):
            mlmodel.predict(sample_input)
