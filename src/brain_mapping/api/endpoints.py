"""
endpoints.py
REST API endpoints for remote workflow execution.
"""
from fastapi import FastAPI, Request
from src.brain_mapping.integration.main_workflow import BrainMappingWorkflow
import numpy as np

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run-workflow")
async def run_workflow(request: Request):
    body = await request.json()
    regions = body.get("regions", 5)
    samples = body.get("samples", 100)
    data = np.random.rand(regions, samples)
    region_labels = {i: f"Region_{i}" for i in range(data)}
    workflow = BrainMappingWorkflow(data, region_labels)
    results = workflow.run()
    return {
        "selected_regions": results["selected_regions"],
        "stats": results["stats"],
        "qc_metrics": results["qc_metrics"]
    }
