"""Setup script for Brain Mapping Toolkit."""

from setuptools import setup, find_packages

setup(
    name="brain-mapping-toolkit",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
