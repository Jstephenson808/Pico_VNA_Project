from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd

from vna.VNA_defaults import DEFAULT_RANDOM_STATE_DEFAULT, DEFAULT_BATCH_PROCESSING_SIZE


class TsfreshBatchedAugmenter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        timeseries_container,
        column_id="id",
        column_sort="time",
        sample_size=100,
        batch_size=DEFAULT_BATCH_PROCESSING_SIZE,
        random_state=DEFAULT_RANDOM_STATE_DEFAULT,
    ):
        self.column_id = column_id
        self.column_sort = column_sort
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.timeseries_container = timeseries_container

    """
                  ┌──────────────┐
                  │ df_ts (full) │◄────────────┐
                  └──────────────┘             │
                                               ▼
            ┌────────────────────────────────────────────┐
            │  TsfreshSafeAugmenter                      │
            │                                            │
            │  .fit(X_train, y_train):                   │
            │   → filter df_ts to X_train.index          │
            │   → select relevant features               │
            │                                            │
            │  .transform(X_test):                       │
            │   → filter df_ts to X_test.index           │
            │   → compute only selected features         │
            └────────────────────────────────────────────┘
    """

    def fit(self, X_train, y_train):
        self.augmenter_ = RelevantFeatureAugmenter(
            column_id=self.column_id, column_sort=self.column_sort
        )
        self.augmenter_.set_params(timeseries_container=self.timeseries_container)

        # Sample from X and y for fitting
        sample_indices = y_train.sample(
            n=self.sample_size, random_state=self.random_state
        ).index
        X_sample = X_train.loc[sample_indices]
        y_sample = y_train.loc[sample_indices]

        self.augmenter_.fit(X_sample, y_sample)
        return self

    def transform(self, X):
        chunks = []
        for i in range(0, len(X), self.batch_size):
            X_batch = X.iloc[i : i + self.batch_size].copy()
            X_aug = self.augmenter_.transform(X_batch)
            chunks.append(X_aug)
        return pd.concat(chunks, axis=0)
