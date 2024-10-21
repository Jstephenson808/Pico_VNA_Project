from abc import ABC, abstractmethod

import pandas as pd
import tsfresh
from tsfresh import defaults
from tsfresh.utilities.dataframe_functions import impute
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
                 drop_cols=[DataFrameCols.LABEL.value],
                 show_warnings:bool=defaults.SHOW_WARNINGS,
                 disable_extraction_progressbar:bool=defaults.DISABLE_PROGRESSBAR
                 ):
        self.parameters = experiment_parameters
        self.feature_extractor = tsfresh.extract_features
        self.feature_selector = tsfresh.select_features
        self.impute_features = impute
        self.n_jobs = n_jobs
        self.drop_cols = drop_cols
        self.ids_per_split = ids_per_split
        self.show_warnings = show_warnings
        self.disable_extraction_progressbar = disable_extraction_progressbar

        self.extracted_features = None
        self.selected_features = None

    def extract_features(self):
        data_frame = self.parameters.s_param_data.data_frame
        combined_df = data_frame.ffill()
        # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
        # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
        data_frame_without_label = combined_df.drop(columns=self.drop_cols)
        if self.ids_per_split > 0:
            split_dfs = self.split_data_frame_into_id_chunks(
                data_frame_without_label, self.ids_per_split
            )
            features_list = [
                self.feature_extractor(
                    df,
                    column_sort=DataFrameCols.TIME.value,
                    column_id=DataFrameCols.ID.value,
                    n_jobs=self.n_jobs,
                    disable_progressbar=self.disable_extraction_progressbar,
                    show_warnings=self.show_warnings
                )
                for df in split_dfs
            ]
            extracted = pd.concat(features_list)
        else:
            extracted = self.feature_extractor(
                data_frame_without_label,
                column_sort=DataFrameCols.TIME.value,
                column_id=DataFrameCols.ID.value,
                n_jobs=self.n_jobs,
                disable_progressbar=self.disable_extraction_progressbar,
                show_warnings=self.show_warnings
            )
        # removes any null values
        extracted = self.impute_features(extracted)
        self.extracted_features = extracted
        return self.extracted_features

    def select_features(self):
        if self.extracted_features is None:
            raise ValueError("Feature extraction has not been run")
        if self.parameters.movement_vector is None:
            raise ValueError("Movement vector is none")
        self.selected_features = self.feature_selector(self.extracted_features,
                                                       self.parameters.movement_vector,
                                                       show_warnings=self.show_warnings)

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

