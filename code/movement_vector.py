import pandas as pd

from VNA_enums import DataFrameCols


class MovementVector:
    def __init__(self, vector:pd.Series=None):
        self.vector = vector


    def create_movement_vector_for_single_data_frame(self, df: pd.DataFrame) -> pd.Series:
        """
        Creates a series which maps each unique id to the associated movement for a results data frame
        Args:
            df:

        Returns:

        """
        movement_dict = {}
        groups = df.groupby([DataFrameCols.ID.value])
        for id_value, id_df in groups:
            movement_dict[id_value[0]] = id_df[DataFrameCols.LABEL.value].values[0]
        self.movement_vector = pd.Series(movement_dict)
        return self.movement_vector
