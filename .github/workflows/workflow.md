# GitHub Actions Workflow Guide

This document describes the CI/CD workflows for the Brain Mapping Toolkit.

## Key Workflows
- test: Runs unit, integration, and coverage tests
- lint: Checks code style and type hints
- build: Builds and uploads package artifacts
- bids-validation: Validates BIDS datasets
- cloud-integration: Tests cloud storage and data workflows
- ml-workflow: Validates ML pipelines
- real-time-analysis: Tests real-time analysis modules
- multi-modal-integration: Validates multi-modal data workflows
- provenance-validation: Checks provenance and audit logging
- property-based-tests: Runs property-based and fuzz tests

## Logs & Artifacts
- All test outputs and logs are saved in the `logs/` folder and uploaded as workflow artifacts
- Coverage reports are uploaded to Codecov

## Maintenance
- Update dependencies in `requirements.txt` and `package.json` regularly
- Refactor code for modularity and maintainability
- Review workflow files after major changes
