from typing import Union, Tuple, Dict
import torch
from torch.utils.flop_counter import FlopCounterMode
from dataclasses import dataclass
from collections import defaultdict


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


class ModelRuntimeAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_stats = defaultdict(LayerStats)

    def analyze(
        self, inp: Union[torch.Tensor, Tuple], batch_time_seconds: float = 1.0
    ) -> ModelStats:
        """
        Analyze both computational and memory requirements of the model.

        Args:
            inp: Input tensor or tuple of input dimensions
            batch_time_seconds: Time taken to process one batch

        Returns:
            ModelStats object containing comprehensive analysis
        """
        self.model.eval()

        # Setup memory tracking hooks
        def count_memory_hook(module, inp, out):
            module_name = module.__class__.__name__

            # Count input reads
            for x in inp:
                if isinstance(x, torch.Tensor):
                    self.layer_stats[module_name].reads += (
                        x.nelement() * x.element_size()
                    )

            # Count output writes
            if isinstance(out, torch.Tensor):
                self.layer_stats[module_name].writes += (
                    out.nelement() * out.element_size()
                )
            elif isinstance(out, tuple):
                for x in out:
                    if isinstance(x, torch.Tensor):
                        self.layer_stats[module_name].writes += (
                            x.nelement() * x.element_size()
                        )

        hooks = []
        for _, module in self.model.named_modules():
            hooks.append(module.register_forward_hook(count_memory_hook))

        flop_counter = FlopCounterMode(display=False, depth=None)

        with flop_counter:
            if isinstance(inp, torch.Tensor):
                self.model(inp)
            elif isinstance(inp, tuple):
                self.model(*inp)
            else:
                self.model(**inp)

        for module_name, flop_dict in flop_counter.get_flop_counts().items():
            # for (op_name, op_flops) in flop_dict.items():
            #     #print(f"{module_name}::{op_name}: {op_flops}")
            #     pass
            total_flops = sum(flop_count for flop_count in flop_dict.values())
            self.layer_stats[module_name].flops = total_flops

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
            layer_stats=dict(self.layer_stats),
        )
