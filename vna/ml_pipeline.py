from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from vna.tsfresh_batched_augmenter import TsfreshBatchedAugmenter

def create_ml_pipeline(timeseries_container):
    """
    Creates a scikit-learn pipeline for feature extraction and classification.

    Args:
        timeseries_container: The container for the time series data that
                              TsfreshBatchedAugmenter will use.

    Returns:
        A scikit-learn Pipeline object.
    """
    return Pipeline([
        (
            'feature_extraction',
            TsfreshBatchedAugmenter(timeseries_container=timeseries_container)
        ),
        (
            'classification',
            RandomForestClassifier(random_state=42)
        )
    ])
