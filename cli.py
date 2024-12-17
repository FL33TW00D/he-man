#!/usr/bin/env python3

from src.models import (
    BlipCaption,
    DepthPro,
    FastVit,
    DetrResnet,
    DistilBert,
    DistilBertANE,
    Mistral7B,
)
from src.shared.runtime_analyzer import ModelRuntimeAnalyzer

import argparse
import csv
import coremltools as ct
from wattkit import Profiler
import time
import torch


def main():
    models_list = [
        # BlipCaption,
        # DepthPro,
        # FastVit,
        # DetrResnet,
        # DistilBert,
        # DistilBertANE,
        Mistral7B,
    ]

    parser = argparse.ArgumentParser(
        description="A CLI tool to work with predefined models."
    )

    model_names_list = [x.name() for x in models_list]
    parser.add_argument(
        "--models",
        nargs="*",
        type=str,
        default=None,
        help=f"The models to use. Choices: {', '.join(model_names_list)}",
    )

    parser.add_argument(
        "--model-iterations",
        type=int,
        default=None,
        help="Override the recommended number of iterations for all the models being run. Must be a positive integer.",
    )

    parser.add_argument(
        "--sample_rate_ms",
        type=int,
        default=50,
        help="Sample rate for the energy profiling.",
    )

    args = parser.parse_args()
    model_names = args.models
    sample_rate = args.sample_rate_ms
    num_samples = 2
    sample_duration = sample_rate * num_samples

    if model_names is not None:
        for m in model_names:
            if m is not None and m not in model_names_list:
                raise Exception(
                    f"Error: '{m}' is not supported. Valid choices are: {', '.join(model_names_list)}."
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
        if model_names is not None and m.name() not in model_names:
            continue

        print(f"Analyzing {m.name()}...")

        model = m()

        print("Starting torch runtime analysis...")
        dummy_input = model.torch_example_input()
        analyzer = ModelRuntimeAnalyzer(model.torch_module())
        torch_runtime_stats = analyzer.analyze(dummy_input)
        print("Finished torch runtime analysis.")
        model.clean_torch_cache()

        for compute_unit, name in [
            (ct.ComputeUnit.CPU_ONLY, "CPU"),
            (ct.ComputeUnit.CPU_AND_GPU, "CPU + GPU"),
            (ct.ComputeUnit.CPU_AND_NE, "CPU + ANE"),
            (ct.ComputeUnit.ALL, "CPU + GPU + ANE"),
        ]:
            print(f"Starting {name} power runtime analysis...")
            with (
                model.setup_run(compute_unit=compute_unit),
                Profiler(
                    sample_duration=sample_duration, num_samples=num_samples
                ) as profiler,
            ):
                model_iterations = model.run(model_iterations=args.model_iterations)

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

    with open("data/output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)


if __name__ == "__main__":
    main()
