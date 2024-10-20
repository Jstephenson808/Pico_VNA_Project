from abc import ABC, abstractmethod

import pandas as pd
import tsfresh
from tsfresh import defaults

from classification_test import ClassificationExperiment, ClassificationExperimentParameters
from VNA_enums import DataFrameCols


class FeatureExtractor(ABC):

    @abstractmethod
    def extract_features(self):
        pass


class TsFreshFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 experiment_parameters: ClassificationExperimentParameters,
                 n_jobs=defaults.N_PROCESSES,
                 ids_per_split=0,
                 drop_cols=[DataFrameCols.LABEL.value]
                 ):
        self.parameters = experiment_parameters
        self.feature_extractor = tsfresh.feature_extraction
        self.n_jobs
        self.extracted_features = None
        self.split_dfs_by_id = None

    def extract_features(self):
        data_frame = self.parameters.s_param_data.data_frame
        combined_df = data_frame.ffill()
        # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
        # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
        data_frame_without_label = combined_df.drop(columns=drop_cols)
        if ids_per_split > 0:
            split_dfs = split_data_frame_into_id_chunks(
                data_frame_without_label, ids_per_split
            )
            features_list = [
                extract_features(
                    df,
                    column_sort=DataFrameCols.TIME.value,
                    column_id=DataFrameCols.ID.value,
                    n_jobs=n_jobs,
                )
                for df in split_dfs
            ]
            extracted = pd.concat(features_list)
        else:
            extracted = extract_features(
                data_frame_without_label,
                column_sort=DataFrameCols.TIME.value,
                column_id=DataFrameCols.ID.value,
                n_jobs=n_jobs,
            )
        extracted = impute(extracted)

    def split_data_frame_into_id_chunks(
            self,
            df: pd.DataFrame,
            ids_per_split: int
    ) -> [pd.DataFrame]:

        # Get the unique IDs
        unique_ids = df[DataFrameCols.ID.value].unique()

        # Initialize a list to store the smaller DataFrames
        split_dfs_by_id = []

        # Split into chunks of n IDs each
        for i in range(0, len(unique_ids), ids_per_split):
            # Get the current chunk of 3 IDs
            chunk_ids = unique_ids[i: i + ids_per_split]

            # Filter the original DataFrame for those IDs
            smaller_df = df[df[DataFrameCols.ID.value].isin(chunk_ids)]

            # Append the resulting DataFrame to the list
            split_dfs_by_id.append(smaller_df)

        return split_dfs_by_id


class TsFreshFeatureExtractionParameters:
    def __init__(self, )