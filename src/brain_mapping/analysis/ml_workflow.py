"""
ML Workflow Manager
==================

Modular, extensible machine learning workflow manager for neuroimaging
analysis. Supports scikit-learn and PyTorch models, plugin integration,
and future extensibility.

Example usage:
    >>> from brain_mapping.analysis.ml_workflow import MLWorkflowManager
    >>> manager = MLWorkflowManager(model_type='sklearn')
    >>> results = manager.automated_analysis(data)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Flags for optional dependencies
SKLEARN_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import sklearn.linear_model
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("brain_mapping.analysis.ml_workflow")


class MLWorkflowError(Exception):
    """Custom exception for MLWorkflowManager errors."""
    pass


class PluginManager:
    """
    Manages plugins for the ML workflow.
    """
    def __init__(self):
        self.plugins: List[Callable] = []

    def register(self, plugin: Callable) -> None:
        """Register a plugin callable to be used in the workflow."""
        self.plugins.append(plugin)
        logger.info(f"Plugin {plugin.__name__} registered.")

    def run_plugins(self, data: Any, **kwargs) -> Any:
        """Run all registered plugins sequentially on the data."""
        for plugin in self.plugins:
            data = plugin(data, **kwargs)
        return data


class MLWorkflowManager:
    """
    Modular machine learning workflow manager for neuroimaging analysis.

    Supports scikit-learn and PyTorch models, plugin integration, and
    extensibility.
    """
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize MLWorkflowManager.

        Parameters
        ----------
        model_type : str, default='auto'
            Type of model ('sklearn', 'torch', or 'custom')
        """
        self.model_type: str = model_type
        self.models = self._load_models()
        self.plugins = []
        logger.info(
            f"MLWorkflowManager initialized with model_type={model_type}"
        )
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        if self.model_type == 'sklearn' and not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for 'sklearn' model_type."
            )
        if self.model_type == 'torch' and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for 'torch' model_type."
            )

    def _load_models(self):
        if self.model_type == 'auto':
            return {'auto': None}
        elif self.model_type == 'sklearn':
            from sklearn.ensemble import RandomForestClassifier
            return {'sklearn': RandomForestClassifier()}
        elif self.model_type == 'torch':
            import torch
            return {'torch': torch.nn.Linear(10, 2)}
        else:
            raise ImportError(f"Unknown model type: {self.model_type}")

    def register_plugin(self, plugin: Callable) -> None:
        """
        Register a plugin callable to be used in the workflow.
        """
        self.plugins.append(plugin)

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            logger.error("Input data must be a numpy ndarray.")
            raise MLWorkflowError("Input data must be a numpy ndarray.")

    def automated_analysis(self, data: np.ndarray):
        """
        Run automated ML analysis pipeline.

        Parameters
        ----------
        data : np.ndarray
            Input data for analysis

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        logger.info("Running automated analysis")
        self._validate_data(data)
        if self.model_type == 'auto':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()
            return {'status': 'success', 'model': clf.fit(data)}
        elif self.model_type == 'sklearn':
            clf = self.models['sklearn']
            return {'status': 'success', 'model': clf.fit(data)}
        elif self.model_type == 'torch':
            model = self.models['torch']
            import torch
            x = torch.tensor(data, dtype=torch.float32)
            return {'status': 'success', 'output': model(x)}
        else:
            return {'status': 'error'}

    def custom_training(
        self, training_data: np.ndarray, labels: np.ndarray
    ):
        """
        Train a custom model on user data.

        Parameters
        ----------
        training_data : np.ndarray
            Training data
        labels : np.ndarray
            Target labels

        Returns
        -------
        Any
            Trained model or None if failed
        """
        logger.info("Running custom training")
        self._validate_data(training_data)
        self._validate_data(labels)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        clf.fit(training_data, labels)
        self.models['custom'] = clf
        return clf

    def predict(self, data: np.ndarray):
        """
        Run inference using the trained model.

        Parameters
        ----------
        data : np.ndarray
            Data to predict on

        Returns
        -------
        np.ndarray or None
            Predictions or None if not available
        """
        logger.info("Running prediction")
        self._validate_data(data)
        if self.model_type == 'sklearn':
            clf = self.models['sklearn']
            return clf.predict(data)
        elif self.model_type == 'auto' and 'custom' in self.models:
            return self.models['custom'].predict(data)
        else:
            raise ImportError("No valid model for prediction")

    def model_interpretation(
        self, model, data: np.ndarray
    ):
        """
        Generate interpretable results from ML models.

        Parameters
        ----------
        model : Any
            Trained model
        data : np.ndarray
            Data to interpret

        Returns
        -------
        Dict[str, Any]
            Interpretation results
        """
        logger.info("Running model interpretation")
        self._validate_data(data)
        try:
            import shap
            explainer = shap.Explainer(model, data)
            return explainer.shap_values(data)
        except ImportError:
            return None

    def as_plugin(self) -> Callable:
        """
        Return a plugin-compatible wrapper for integration with pipelines.

        Returns
        -------
        object
            Plugin-compatible callable
        """
        def plugin_fn(data, **kwargs):
            for plugin in self.plugins:
                data = plugin(data, **kwargs)
            return data
        return plugin_fn

def run_ml_pipeline(data, labels=None):
    """Run a basic ML pipeline (example: fit RandomForest if labels provided)."""
    from sklearn.ensemble import RandomForestClassifier
    if labels is not None:
        clf = RandomForestClassifier()
        clf.fit(data, labels)
        return clf
    else:
        return None

def extract_features(data):
    """Extract basic features (mean, std) from data."""
    import numpy as np
    return {'mean': np.mean(data), 'std': np.std(data)}

def interpret_model(model, data):
    """Interpret model using SHAP (if available)."""
    try:
        import shap
        explainer = shap.Explainer(model, data)
        return explainer.shap_values(data)
    except ImportError:
        return None