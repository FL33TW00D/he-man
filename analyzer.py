from typing import Union, Tuple, Dict 
import torch
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import resnet18
from dataclasses import dataclass
from torchinfo import summary
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class LayerStats:
    flops: int = 0
    reads: int = 0
    writes: int = 0
    
    @property
    def total_memory(self) -> int:
        return self.reads + self.writes
    
    @property
    def arithmetic_intensity(self) -> float:
        """Calculate arithmetic intensity (FLOPs per byte of memory access)"""
        return self.flops / self.total_memory if self.total_memory > 0 else 0

@dataclass
class ModelStats:
    total_flops: int
    total_reads: int
    total_writes: int
    bandwidth_gb_s: float
    gflops: float
    layer_stats: Dict[str, LayerStats]
    
    @property
    def total_memory(self) -> int:
        return self.total_reads + self.total_writes
    
    @property
    def overall_arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_memory if self.total_memory > 0 else 0

class ModelAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_stats = defaultdict(LayerStats)
        
    def analyze(self, 
                inp: Union[torch.Tensor, Tuple],
                batch_time_seconds: float = 1.0) -> ModelStats:
        """
        Analyze both computational and memory requirements of the model.
        
        Args:
            inp: Input tensor or tuple of input dimensions
            batch_time_seconds: Time taken to process one batch
            
        Returns:
            ModelStats object containing comprehensive analysis
        """
        self.model.eval()
        
        # Reset statistics
        self.layer_stats.clear()
        
        # Setup memory tracking hooks
        def count_memory_hook(module, inp, out):
            module_name = module.__class__.__name__
            
            # Count input reads
            for x in inp:
                if isinstance(x, torch.Tensor):
                    self.layer_stats[module_name].reads += x.nelement() * x.element_size()
            
            # Count output writes
            if isinstance(out, torch.Tensor):
                self.layer_stats[module_name].writes += out.nelement() * out.element_size()
            elif isinstance(out, tuple):
                for x in out:
                    if isinstance(x, torch.Tensor):
                        self.layer_stats[module_name].writes += x.nelement() * x.element_size()
        
        hooks = []
        for _, module in self.model.named_modules():
            hooks.append(module.register_forward_hook(count_memory_hook))
        
        inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
        flop_counter = FlopCounterMode(mods=self.model, display=True, depth=None)
        
        with flop_counter:
            self.model(inp)
        
        for module_name, flop_dict in flop_counter.get_flop_counts().items():
            print(f"MOD: {module_name} FLOPS: {flop_dict}")
            for (op_name, op_flops) in flop_dict.items():
                #print(f"{module_name}::{op_name}: {op_flops}")
                pass

            self.layer_stats[module_name].flops = flop_dict 
        
        total_flops = flop_counter.get_total_flops()
        total_reads = sum(stats.reads for stats in self.layer_stats.values())
        total_writes = sum(stats.writes for stats in self.layer_stats.values())
        
        bandwidth_gb_s = ((total_reads + total_writes) / (1024**3)) / batch_time_seconds
        gflops = total_flops / 1e9
        
        for hook in hooks:
            hook.remove()
        
        return ModelStats(
            total_flops=total_flops,
            total_reads=total_reads,
            total_writes=total_writes,
            bandwidth_gb_s=bandwidth_gb_s,
            gflops=gflops,
            layer_stats=dict(self.layer_stats)
        )

def plot_per_module_stats(stats: ModelStats):
    """Plot per module FLOPs and memory access statistics"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    module_names = list(stats.layer_stats.keys())
    flops = [stats.layer_stats[name].flops for name in module_names]
    reads = [stats.layer_stats[name].reads for name in module_names]
    writes = [stats.layer_stats[name].writes for name in module_names]

    print("Module Names: ", module_names)
    print("FLOPS: ", flops)
    print("Reads: ", reads)
    print("Writes: ", writes)
    
    ax[0].barh(module_names, flops, color='skyblue', label='FLOPs')
    ax[0].barh(module_names, reads, color='orange', label='Memory Reads')
    ax[0].barh(module_names, writes, color='green', label='Memory Writes')
    ax[0].set_xlabel("Count")
    ax[0].set_title("Per Module FLOPs and Memory Access")
    ax[0].legend()
    
    ai = [stats.layer_stats[name].arithmetic_intensity for name in module_names]
    ax[1].barh(module_names, ai, color='salmon')
    ax[1].set_xlabel("FLOPs/byte")
    ax[1].set_title("Per Module Arithmetic Intensity")
    
    plt.tight_layout()
    plt.show()

def format_bytes(bytes: int) -> str:
    """Format bytes into human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"


def validate_model(model: torch.nn.Module):
    model.eval()
    import requests
    from PIL import Image
    from torchvision.models import ResNet18_Weights
    from io import BytesIO
    import json

    response = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
    labels = json.loads(response.content)

    # hummingbird == 94
    response = requests.get("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/refs/heads/master/n01833805_hummingbird.JPEG")
    img = Image.open(BytesIO(response.content))
    transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    with torch.no_grad():
        result = model(transforms(img).unsqueeze(0))
    argmax = result.argmax()
    assert argmax.item() == 94
    label = labels[argmax]
    assert label == "hummingbird"
    print(f"Validation image is a {label} 🐤!")


# Example usage
if __name__ == "__main__":
    model = resnet18(weights='IMAGENET1K_V1')
    model = model.eval()
    summary(model, input_size=(1,3,224,224))

    validate_model(model)

    analyzer = ModelAnalyzer(model)
    stats = analyzer.analyze((1, 3, 224, 224), batch_time_seconds=0.016)  # ~60 FPS

    print("\n STATS: ", stats)
    
    print(f"\nModel Analysis Results:")
    print(f"----------------------")
    print(f"Total FLOPs: {stats.gflops:.2f} GFLOPs")
    print(f"Memory Reads: {format_bytes(stats.total_reads)}")
    print(f"Memory Writes: {format_bytes(stats.total_writes)}")
    print(f"Required Memory Bandwidth: {stats.bandwidth_gb_s:.2f} GB/s")
    print(f"Overall Arithmetic Intensity: {stats.overall_arithmetic_intensity:.2f} FLOPs/byte")

    plot_per_module_stats(stats)
