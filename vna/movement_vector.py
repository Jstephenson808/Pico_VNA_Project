from typing import TYPE_CHECKING
import pandas as pd
from VNA_enums import DataFrameCols

# Assuming DataFrameCols is defined elsewhere and imported properly
if TYPE_CHECKING:
    from __main__ import MovementVector


class MovementVector:

    @staticmethod
    def create_movement_vector_for_single_data_frame(df: pd.DataFrame) -> "MovementVector":
        """
        Creates a series which maps each unique id to the associated movement for a results data frame
        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            MovementVector: A MovementVector instance with the created movement vector.
        """
        movement_dict = {}
        groups = df.groupby([DataFrameCols.ID.value])
        for id_value, id_df in groups:
            movement_dict[id_value[0]] = id_df[DataFrameCols.LABEL.value].values[0]
        movement_vector = MovementVector(pd.Series(movement_dict))
        return movement_vector

    def __init__(self, vector:pd.Series=None):
        self.movement_vector = vector



