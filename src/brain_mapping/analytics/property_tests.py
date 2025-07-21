"""
Property-based and fuzz tests for analytics modules.
"""
from hypothesis import given, strategies as st
from brain_mapping.analytics.advanced_analytics import AdvancedAnalytics


@given(
    st.lists(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6),
            min_size=3, max_size=3
        ),
        min_size=3, max_size=10
    )
)
def test_pca_property(data):
    aa = AdvancedAnalytics()
    result = aa.run_pca(data, n_components=2)
    assert result is not None
    assert len(result) == len(data)
