import numpy as np
from PIL import Image
import coremltools as ct

from typing import Dict, Any

def random_input_dict(model: ct.MLModel) -> Dict[str, Any]:
    inputs = {}
    
    try:
        spec = model.get_spec()
    except AttributeError:
        spec = model.spec if hasattr(model, 'spec') else model
    
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