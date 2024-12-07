from .model import Model
import torch

class Llama321B(Model):
    def name():
        return "meta-llama/Llama-3.2-1B"
    
    def __init__(self):
        super().__init__()

    def torch_module() -> torch.nn.Module:
        pass