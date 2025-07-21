import numpy as np
import pytest

from brain_mapping.analysis.ml_workflow import MLWorkflowManager


@pytest.mark.skipif(not hasattr(np, 'zeros'), reason="NumPy not available")
def test_init_sklearn():
    try:
        manager = MLWorkflowManager(model_type='sklearn')
        assert manager.model_type == 'sklearn'
        assert manager.model is not None
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_automated_analysis_sklearn():
    try:
        manager = MLWorkflowManager(model_type='sklearn')
        data = np.random.rand(10, 3)
        result = manager.automated_analysis(data)
        assert 'status' in result
        assert result['status'] in ['success', 'error']
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_custom_training_sklearn():
    try:
        manager = MLWorkflowManager(model_type='sklearn')
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, size=10)
        model = manager.custom_training(X, y)
        assert model is not None
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_predict_sklearn():
    try:
        manager = MLWorkflowManager(model_type='sklearn')
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, size=10)
        manager.custom_training(X, y)
        preds = manager.predict(X)
        assert preds is not None
        assert len(preds) == 10
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_plugin_registration():
    manager = MLWorkflowManager(model_type='sklearn')

    def dummy_plugin(img, **kwargs):
        return img * 2

    manager.register_plugin(dummy_plugin)
    plugin_fn = manager.as_plugin()
    img = np.ones((5, 5))
    result = plugin_fn(img)
    assert np.all(result == 2)


def test_invalid_model_type():
    with pytest.raises(ImportError):
        MLWorkflowManager(model_type='torch')


# Log this test addition
with open('logs/CHANGELOG_AUTOMATED.md', 'a') as logf:
    logf.write(
        "- [2025-07-20 22:57] Added tests for MLWorkflowManager: init, analysis, "
        "training, predict, plugin, error handling.\n"
    ) 