#!/usr/bin/env python3

from src.models.blip_caption import BlipCaption
from src.models.depth_pro import DepthPro
from src.models.fast_vit import FastVit
from src.models.llama_3_2_1B import Llama321B

from src.shared.runtime_analyzer import ModelRuntimeAnalyzer

import argparse
import sys

from wattkit import Profiler

import coremltools as ct

def main():
    models_list = [BlipCaption, DepthPro, FastVit, Llama321B]

    parser = argparse.ArgumentParser(
        description="A CLI tool to work with predefined models."
    )
    
    model_names_list = [x.name() for x in models_list]
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f"The model to use. Choices: {', '.join(model_names_list)}"
    )

    args = parser.parse_args()
    model_name = args.model

    if model_name is not None and model_name not in model_names_list:
        raise Exception(f"Error: '{args.model}' is not supported. Valid choices are: {', '.join(model_names_list)}.")


    for m in models_list:
        if model_name is not None and model_name != m.name():
            continue

        print(f"Analyzing {m.name()}...")

        model = m()

        print("Obtaining torch model...")
        module = model.torch_module()
        print("Finished obtaining torch module.")

        print("Starting torch runtime analysis...")
        dummy_input = model.torch_example_input()
        analyzer = ModelRuntimeAnalyzer(module)
        torch_runtime_stats = analyzer.analyze(dummy_input)
        print(torch_runtime_stats)
        print("Finished torch runtime analysis.")

        # coreml stuff
        print("Obtaining Core ML model...")
        ct_model = model.coreml_model()
        ct_model.save("ct_model.mlpackage")
        print("Finished obtaining Core ML model.")

        # pip install git+https://github.com/FL33TW00D/wattkit.git@master#subdirectory=bindings/python
        # .venv/bin/python cli.py --model apple/DepthPro-mixin
        coreml_dummy_input = model.coreml_example_input()

        model_iterations = 1
        for compute_unit, name in [(ct.ComputeUnit.CPU_ONLY, "cpu"), (ct.ComputeUnit.CPU_AND_GPU, "gpu"), (ct.ComputeUnit.CPU_AND_NE, "ane")]:
            print(f"Starting {name} power runtime analysis...")
            # I'm so sorry, It must happen https://github.com/apple/coremltools/issues/1849
            ct_model = ct.models.MLModel("ct_model.mlpackage", compute_units=compute_unit)
            ct_model.predict(coreml_dummy_input) # Once before to "warm up" hardware
            with Profiler(sample_duration=100, num_samples=2) as profiler:
                for i in range(model_iterations):
                    ct_model.predict(coreml_dummy_input)
            print(profiler.get_profile())
            print(f"Finished {name} power runtime analysis.")

        print(f"Finished analyzing {args.model}.")


if __name__ == '__main__':
    main()
