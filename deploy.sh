#!/bin/bash
# Deployment script for cloud platforms
set -e

echo "Building Docker image..."
docker build -t brain-mapping:latest .

echo "Running container..."
docker run --rm -it -v $(pwd)/logs:/app/logs brain-mapping:latest
