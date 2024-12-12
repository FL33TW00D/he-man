#!/usr/bin/env python3

from src.models import (
    BlipCaption,
    DepthPro,
    FastVit,
    DetrResnet,
    DistilBert,
    DistilBertANE,
)
from src.shared.runtime_analyzer import ModelRuntimeAnalyzer

import argparse
import csv
import coremltools as ct
from wattkit import Profiler
from shutil import copytree
import time
import torch

def main():
    models_list = [
        #BlipCaption,
        #DepthPro,
        #FastVit,
        #DetrResnet,
        DistilBert,
        DistilBertANE,
    ]

    parser = argparse.ArgumentParser(
        description="A CLI tool to work with predefined models."
    )

    model_names_list = [x.name() for x in models_list]
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"The model to use. Choices: {', '.join(model_names_list)}",
    )

    parser.add_argument(
        "--model-iterations",
        type=int,
        default=None,
        help="The number of iterations for the model. Must be a positive integer.",
    )

    parser.add_argument(
        "--sample_rate_ms",
        type=int,
        default=50,
        help="Sample rate for the energy profiling.",
    )

    args = parser.parse_args()
    model_name = args.model
    sample_rate = args.sample_rate_ms
    num_samples = 2
    sample_duration = sample_rate * num_samples

    if model_name is not None and model_name not in model_names_list:
        raise Exception(
            f"Error: '{args.model}' is not supported. Valid choices are: {', '.join(model_names_list)}."
        )

    torch.set_grad_enabled(False)

    output_data = [
        [
        "Model",
        "Compute Unit",
        "Total FLOPs",
        "Total Reads",
        "Total Writes",
        "Model Iterations",
        "Total CPU Energy",
        "Total GPU Energy",
        "Total ANE Energy",
        "Average CPU Power",
        "Average GPU Power",
        "Average ANE Power",
        "Total Duration",
        ]
    ]
    start_time = time.time()
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
        print("Finished torch runtime analysis.")

        # coreml stuff
        print("Obtaining Core ML model...")
        ct_model = model.coreml_model()
        compiled_model_path = ct_model.get_compiled_model_path()
        copytree(compiled_model_path, "ct_model.mlmodelc", dirs_exist_ok=True)
        print("Finished obtaining Core ML model.")

        coreml_dummy_input = model.coreml_example_input()

        for compute_unit, name in [
            (ct.ComputeUnit.CPU_ONLY, "CPU"),
            (ct.ComputeUnit.CPU_AND_GPU, "CPU + GPU"),
            (ct.ComputeUnit.CPU_AND_NE, "CPU + ANE"),
            (ct.ComputeUnit.ALL, "CPU + GPU + ANE"),
        ]:
            print(f"Starting {name} power runtime analysis...")
            ct_model = ct.models.CompiledMLModel(
                "ct_model.mlmodelc", compute_units=compute_unit
            )
            model_iterations = model.recommended_iterations() if args.model_iterations is None else args.model_iterations
            ct_model.predict(coreml_dummy_input)  # Once before to "warm up" hardware
            with Profiler(
                sample_duration=sample_duration, num_samples=num_samples
            ) as profiler:
                for _ in range(model_iterations):
                    ct_model.predict(coreml_dummy_input)
            profile = profiler.get_profile()
            outp = [
                str(x)
                for x in [
                    m.name(),
                    name,
                    torch_runtime_stats.total_flops,
                    torch_runtime_stats.total_reads,
                    torch_runtime_stats.total_writes,
                    model_iterations,
                    profile.total_cpu_energy,
                    profile.total_gpu_energy,
                    profile.total_ane_energy,
                    profile.average_cpu_power,
                    profile.average_gpu_power,
                    profile.average_ane_power,
                    profile.total_duration,
                ]
            ]
            print("     ", ",".join(outp))
            output_data.append(outp)
            print(f"Finished {name} power runtime analysis.")

        print(f"Finished analyzing {m.name()}.")

    print(f"Total analysis time: {time.time() - start_time} seconds.")

    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)


if __name__ == "__main__":
    main()
