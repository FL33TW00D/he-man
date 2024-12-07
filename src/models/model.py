from abc import abstractmethod
from typing import Dict, List, Union, Any

import torch
import coremltools as ct
from PIL import Image
import numpy as np

class Model():
    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def torch_module(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def torch_example_input(self) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def coreml_model(self) -> ct.models.MLModel:
        pass

    def coreml_example_input(self) -> Dict[str, Any]:
        inputs = {}

        ct_model = self.coreml_model()
        
        try:
            spec = ct_model.get_spec()
        except AttributeError:
            spec = ct_model.spec if hasattr(ct_model, 'spec') else ct_model
        
        for input_desc in spec.description.input:
            input_name = input_desc.name
            
            if hasattr(input_desc, 'type'):
                if input_desc.type.HasField('imageType'):
                    image_type = input_desc.type.imageType
                    height = image_type.height
                    width = image_type.width
                    
                    if image_type.colorSpace == 0: # GRAYSCALE
                        channels = 1
                    else: # Default to 3
                        channels = 3

                    noise_array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
                    inputs[input_name] = Image.fromarray(noise_array)
                elif input_desc.type.HasField('multiArrayType'):
                    shape = tuple(input_desc.type.multiArrayType.shape)
                    inputs[input_name] = np.random.randn(*shape)
                else:
                    raise Exception(f"Could not determine input type for {input_name}")
            else:
                raise Exception(f"Could not determine input type for {input_name}")
        
        return inputs
