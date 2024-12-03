# HE-MAN

> **Mission Statement** \
> Understanding the power usage and "compute profile" of modern transformers suited for the edge.
> Fundamental Q: given a model, what hardware requirements do I need (FLOPS + GB/s bandwidth) to run it

## What is a compute profile?

> **Compute Profile** \
> The compute profile of a model is the pattern of dataflow and computation for an entire model (layer by layer).

Some important metrics are:
- Arithmetic Intensity (FLOP/B)
- Energy Usage (TFLOP/J)
- Layer by Layer parameter usage
- Layer by Layer MACs

Relevant resources:
- https://arxiv.org/pdf/2109.14320
- https://arxiv.org/pdf/2206.15472

## What can we do to improve performance?

What optimizations can be done to make the models more suited for ANE? 

- Layer reordering
- Data layout reordering
- Linear -> Conv2d

Relevant resources:
- https://github.com/apple/ml-ane-transformers
- https://machinelearning.apple.com/research/neural-engine-transformers

## Best models to investigate?

Some of the best models to investigate might include:
- SmolLM
- Moondream
- Whisper
- TTS? 
- Pose estimation?
- Segmentation?

## Key figures

![image](https://github.com/user-attachments/assets/afdd247e-9548-4da3-b872-7965fc5e1948)
![image](https://github.com/user-attachments/assets/4a205e26-925d-41c2-bedd-d9e60cd45bbf)
![image](https://github.com/user-attachments/assets/836d5733-93ec-41d8-9f61-6b99b897a0db)
![image](https://github.com/user-attachments/assets/ebe74a99-5860-44bc-bd64-90a0e79bd7d9)


## Tooling

- CUDA memory stats: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
- FlopCounterMode
- Calculating MACs and memory reads by hand
