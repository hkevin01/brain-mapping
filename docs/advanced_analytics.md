# Advanced Analytics Guide

## Overview
This guide covers advanced analytics features such as PCA and ML workflows in the brain-mapping toolkit.

## Modules
- `src/brain_mapping/analytics/advanced_analytics.py`: PCA and dimensionality reduction
- `src/brain_mapping/analysis/ml_workflow_manager.py`: ML workflow management

## Example Usage
```python
from brain_mapping.analytics.advanced_analytics import AdvancedAnalytics
analytics = AdvancedAnalytics()
data = ... # your data
transformed, variance = analytics.run_pca(data)
```
