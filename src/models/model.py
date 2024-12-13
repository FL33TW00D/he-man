from abc import abstractmethod
from typing import Dict, List, Union, Any

import torch
import coremltools as ct
from PIL import Image
import numpy as np
import warnings

class ModelContextManager:
    def __init__(self, model):
        self.inside_context = False
        self.model = model
    
    def __enter__(self):
        self.inside_context = True

    def __exit__(self, *_):
        self.inside_context = False

class Model:
    def __init__(self):
        self.cached_torch_trace = None
        self.cached_coreml_input = None
        self.cached_coreml_model = None
        self.context_manager = None

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
        if self.cached_coreml_input:
            return self.cached_coreml_input
        
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

        self.cached_coreml_input = inputs

        return inputs
    
    def setup_run(
        self, 
        compute_unit=ct.ComputeUnit.ALL
    ) -> ModelContextManager:
        coreml_dummy_input = self.coreml_example_input()
        compiled_model_path = self.coreml_model().get_compiled_model_path()
        ct_model = ct.models.CompiledMLModel(
            compiled_model_path, 
            compute_units=compute_unit
        )
        ct_model.predict(coreml_dummy_input)
        self.context_manager = ModelContextManager(ct_model)
        return self.context_manager
    
    def run(
        self,
        model_iterations=None
    ) -> Dict:
        if self.context_manager is None or not self.context_manager.inside_context:
            print("Error: `run` must be called inside `setup_run` context")
            raise Exception("`run` must be called inside `setup_run` context")
        if model_iterations is None:
            model_iterations = self.recommended_iterations()
        coreml_dummy_input = self.coreml_example_input()

        ct_model = self.context_manager.model
        for _ in range(model_iterations):
            ct_model.predict(coreml_dummy_input)

        return model_iterations




