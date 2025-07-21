"""
Example script to run the integrated brain mapping workflow
with visualization and QC.
"""
import numpy as np
from src.brain_mapping.integration.main_workflow import BrainMappingWorkflow


def main():
    # Example data and region labels
    data = np.random.rand(5, 100)  # 5 regions, 100 samples each
    region_labels = {i: f"Region_{i}" for i in range(data.shape[0])}
    workflow = BrainMappingWorkflow(data, region_labels)
    results = workflow.run()
    print("Selected regions:", results["selected_regions"])
    print("Region stats:", results["stats"])
    print("QC metrics:", results["qc_metrics"])
    print("See logs/ for output files.")


if __name__ == "__main__":
    main()
