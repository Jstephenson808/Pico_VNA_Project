from abc import ABCMeta, abstractmethod, ABC
from typing import TYPE_CHECKING, Generic
import pandas as pd
from vna.VNA_enums import DataFrameCols
from vna.VNA_types import (
    DataFrameType,
    MovementVectorOrSubClass,
    SParamDataOrSubClass,
    Series,
)

from vna.s_parameter_data import SParameterData, SParameterDataPandas

# Assuming DataFrameCols is defined elsewhere and imported properly
if TYPE_CHECKING:
    from __main__ import MovementVectorPandas


class MovementVector(Generic[Series], ABC):
    def __init__(self, vector: Series, label: str):
        self.movement_vector: Series = vector
        self.label: str = label

    @classmethod
    @abstractmethod
    def create_movement_vector_for_single_data_frame(
        cls,
        s_parameter_data: SParamDataOrSubClass,
    ) -> MovementVectorOrSubClass:
        """
        Creates a series which maps each unique id to the associated movement for a results data frame
        Args:
            s_parameter_data: the input dataframe object, whatever the input is the output will be of the same type of data back end

        Returns:
            MovementVector: A MovementVector instance with the created movement vector.
        """
        pass


class MovementVectorPandas(MovementVector):

    @classmethod
    def create_movement_vector_for_single_data_frame(
        cls,
        s_parameter_data_pandas: SParameterDataPandas,
    ) -> MovementVectorPandas:
        """
        Creates a series which maps each unique id to the associated movement for a results data frame
        Args:
            s_parameter_data_pandas: input pandas data container
        Returns:
            MovementVectorPandas: A MovementVector instance with the created movement vector.
        """
        movement_dict = {}
        groups = s_parameter_data_pandas.data_frame.groupby([DataFrameCols.ID.value])
        for id_value, id_df in groups:
            movement_dict[id_value[0]] = id_df[DataFrameCols.LABEL.value].values[0]
        movement_vector = MovementVectorPandas(pd.Series(movement_dict))
        return movement_vector
