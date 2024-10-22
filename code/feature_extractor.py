from abc import ABC, abstractmethod

import pandas as pd
import tsfresh
from tsfresh.utilities.dataframe_functions import impute
from classification_test import ClassificationExperimentParameters
from VNA_enums import DataFrameCols
from feature_extraction_parameters import FeatureExtractionParameters


class FeatureExtractor(ABC):

    @abstractmethod
    def extract_features(self):
        pass


class TsFreshFeatureExtractor(FeatureExtractor):

    def __init__(self,
                 experiment_parameters: ClassificationExperimentParameters,
                 feature_extraction_parameters: FeatureExtractionParameters):
        self.feature_extraction_parameters = feature_extraction_parameters
        self.parameters = experiment_parameters
        self.feature_extractor = tsfresh.extract_features
        self.feature_selector = tsfresh.select_features
        self.impute_features = impute

        self.extracted_features = None
        self.selected_features = None

    def extract_features(self):
        data_frame = self.parameters.s_param_data.data_frame
        combined_df = data_frame.ffill()
        # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
        # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
        data_frame_without_label = combined_df.drop(columns=self.feature_extraction_parameters.drop_cols)
        if self.feature_extraction_parameters.ids_per_split > 0:
            split_dfs = self.split_data_frame_into_id_chunks(
                data_frame_without_label, self.feature_extraction_parameters.ids_per_split
            )
            features_list = [
                self.feature_extractor(
                    df,
                    column_sort=DataFrameCols.TIME.value,
                    column_id=DataFrameCols.ID.value,
                    n_jobs=self.feature_extraction_parameters.n_jobs,
                    disable_progressbar=self.feature_extraction_parameters.disable_extraction_progressbar,
                    show_warnings=self.feature_extraction_parameters.show_warnings
                )
                for df in split_dfs
            ]
            extracted = pd.concat(features_list)
        else:
            extracted = self.feature_extractor(
                data_frame_without_label,
                column_sort=DataFrameCols.TIME.value,
                column_id=DataFrameCols.ID.value,
                n_jobs=self.feature_extraction_parameters.n_jobs,
                disable_progressbar=self.feature_extraction_parameters.disable_extraction_progressbar,
                show_warnings=self.feature_extraction_parameters.show_warnings
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
                                                       show_warnings=self.feature_extraction_parameters.show_warnings)

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

