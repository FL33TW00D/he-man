from .model import Model
import torch

class BlipCaption(Model):
    def name():
        return "Salesforce/blip-image-captioning-base"
    
    def __init__(self):
        super().__init__()

    def torch_module() -> torch.nn.Module:
        pass