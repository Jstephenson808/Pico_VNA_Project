from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from pandas import DataFrame
import tsfresh
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from classification_experiment_parameters import ClassificationExperimentParameters
from VNA_enums import DataFrameCols
from s_parameter_data import SParameterData
from feature_extraction_parameters import FeatureExtractionParameters


class FullOrSelectedFeatures(Enum):
    Full_Features = "Full_Features"
    Selected_Features = "Selected_Features"


class ExtractedFeatures:

    def __init__(
        self,
        extracted_features: DataFrame,
        full_or_selected_features: FullOrSelectedFeatures,
    ):
        self.extracted_features: DataFrame = extracted_features
        self.full_or_selected_features: FullOrSelectedFeatures = (
            full_or_selected_features
        )


class FeatureExtractor(ABC):

    @abstractmethod
    def extract_features(self, s_parameter_data: SParameterData) -> ExtractedFeatures:
        pass


class TsFreshFeatureExtractor(FeatureExtractor):

    def __init__(self, feature_extraction_parameters: FeatureExtractionParameters):
        self.feature_extraction_parameters: FeatureExtractionParameters = (
            feature_extraction_parameters
        )
        self.feature_extractor = tsfresh.extract_features
        self.feature_selector = tsfresh.select_features
        self.impute_features = impute

        self.extracted_features: ExtractedFeatures | None = None
        self.selected_features: ExtractedFeatures | None = None

    def extract_features(self, s_parameter_data: SParameterData) -> ExtractedFeatures:
        self.feature_extraction_only(s_parameter_data)
        if self.feature_extraction_parameters.select_features:
            self.select_features()
            return self.selected_features
        else:
            return self.extracted_features

    def feature_extraction_only(
        self, s_param_data: SParameterData
    ) -> ExtractedFeatures:
        data_frame: DataFrame = s_param_data.get_full_data_frame()
        combined_df = data_frame.ffill()
        # s_params_mapping = {s.value:index+1 for index, s in enumerate(SParam)}
        # full_data_frame[DataFrameCols.S_PARAMETER.value].map({s.value: index for index, s in enumerate(SParam)})
        data_frame_without_label = combined_df.drop(
            columns=self.feature_extraction_parameters.drop_cols
        )
        if self.feature_extraction_parameters.ids_per_split > 0:
            split_dfs = self.split_data_frame_into_id_chunks(
                data_frame_without_label,
                self.feature_extraction_parameters.ids_per_split,
            )
            features_list = [
                self.feature_extractor(
                    df,
                    column_sort=DataFrameCols.TIME.value,
                    column_id=DataFrameCols.ID.value,
                    n_jobs=self.feature_extraction_parameters.n_jobs,
                    disable_progressbar=self.feature_extraction_parameters.disable_extraction_progressbar,
                    show_warnings=self.feature_extraction_parameters.show_warnings,
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
                show_warnings=self.feature_extraction_parameters.show_warnings,
            )
        # removes any null values
        extracted = self.impute_features(extracted)
        extracted = ExtractedFeatures(extracted, FullOrSelectedFeatures.Full_Features)
        self.extracted_features = extracted
        return self.extracted_features

    def select_features(self):
        if self.extracted_features is None:
            raise ValueError("Feature extraction has not been run")
        if self.parameters.movement_vector is None:
            raise ValueError("Movement vector is none")
        self.selected_features = self.feature_selector(
            self.extracted_features,
            self.parameters.movement_vector,
            show_warnings=self.feature_extraction_parameters.show_warnings,
        )

    def split_data_frame_into_id_chunks(
        self, df: DataFrame, ids_per_split: int
    ) -> [DataFrame]:

        # Get the unique IDs
        unique_ids = df[DataFrameCols.ID.value].unique()

        # Initialize a list to store the smaller DataFrames
        split_dfs_by_id = []

        # Split into chunks of n IDs each
        for i in range(0, len(unique_ids), ids_per_split):
            # Get the current chunk of 3 IDs
            chunk_ids = unique_ids[i : i + ids_per_split]

            # Filter the original DataFrame for those IDs
            smaller_df = df[df[DataFrameCols.ID.value].isin(chunk_ids)]

            # Append the resulting DataFrame to the list
            split_dfs_by_id.append(smaller_df)

        return split_dfs_by_id
