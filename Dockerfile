# Dockerfile for brain-mapping toolkit

# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libxss1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt pyproject.toml setup.py ./
COPY src/brain_mapping/_version.py src/brain_mapping/_version.py

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Copy source code
COPY src/ src/
COPY docs/ docs/
COPY tests/ tests/

# Create data and output directories
RUN mkdir -p /app/data /app/output

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MPLBACKEND=Agg

# Expose port for Jupyter (if needed)
EXPOSE 8888

# Create non-root user
RUN useradd -m -u 1000 brainmapper
USER brainmapper

# Default command
CMD ["python", "src/brain_mapping/cli/main_cli.py"]
