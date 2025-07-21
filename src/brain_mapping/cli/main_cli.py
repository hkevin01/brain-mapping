"""
main_cli.py
Simple CLI for running the brain mapping workflow.
"""
import argparse
from src.brain_mapping.integration.main_workflow import BrainMappingWorkflow
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run brain mapping workflow"
    )
    parser.add_argument(
        "--regions", type=int, default=5,
        help="Number of regions"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Samples per region"
    )
    parser.add_argument(
        "--output", type=str, default="logs/cli_output.txt",
        help="Path to output log file"
    )
    args = parser.parse_args()
    data = np.random.rand(args.regions, args.samples)
    region_labels = {i: f"Region_{i}" for i in range(data.shape[0])}
    workflow = BrainMappingWorkflow(data, region_labels)
    results = workflow.run()
    print("Selected regions:", results["selected_regions"])
    print("Region stats:", results["stats"])
    print("QC metrics:", results["qc_metrics"])
    with open(args.output, "w") as f:
        f.write(str(results))
    print(f"Results saved to {args.output}")
    print("See logs/ for output files.")


if __name__ == "__main__":
    main()
