# Cloud Deployment Guide

## Overview
This guide describes how to deploy the brain-mapping toolkit on AWS and GCP using Docker and cloud storage integration.

## Steps
1. Build Docker image using the provided Dockerfile.
2. Configure cloud credentials in `config/cloud_config.yaml`.
3. Use `deploy.sh` to run the container and connect to cloud storage.
4. Refer to `src/brain_mapping/cloud/cloud_processor.py` for upload and processing utilities.

## Example Usage
```bash
docker build -t brain-mapping:latest .
docker run --rm -it -v $(pwd)/logs:/app/logs brain-mapping:latest
```
