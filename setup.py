"""Setup script for Brain Mapping Toolkit."""

from setuptools import setup, find_packages

setup(
    name="brain-mapping",
    version="0.1.0",
    description="Advanced brain mapping toolkit with visualization and QC",
    author="hkevin01",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "hypothesis",
        "mypy",
        "flake8",
        "black",
        "boto3",
        "google-cloud-storage",
        "scikit-learn",
        "torch",
        "tensorflow",
        "shap",
        "mne",
        "neo",
    ],
    extras_require={
        "dev": ["pytest", "hypothesis", "mypy", "flake8", "black"],
    },
)
