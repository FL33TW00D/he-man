from .model import Model
import torch

class FastVit(Model):
    def name():
        return "timm/fastvit_ma36.apple_in1k"
    
    def __init__(self):
        super().__init__()

    def torch_module() -> torch.nn.Module:
        pass