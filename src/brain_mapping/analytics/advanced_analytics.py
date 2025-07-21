"""
advanced_analytics.py
Advanced analytics functions for brain mapping toolkit.
"""
import numpy as np
from sklearn.decomposition import PCA


class AdvancedAnalytics:
    """Advanced analytics for brain mapping data."""
    def run_pca(self, data, n_components=2):
        """Run PCA on input data."""
        data = np.array(data)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return transformed, pca.explained_variance_ratio_
