from __future__ import annotations

from typing import TypeVar

import pandas as pd
import polars as pl
from pandas.core.groupby import DataFrameGroupBy

DataFrameType = TypeVar("DataFrameType", pd.DataFrame, pl.DataFrame)
Series = TypeVar("Series", pd.Series, pl.Series)
GroupBy = TypeVar("GroupBy", DataFrameGroupBy, pl.DataFrameGroupBy)
SParamDataOrSubClass = TypeVar("SParamDataOrSubClass", bound="SParameterData")
MovementVectorOrSubClass = TypeVar("MovementVectorOrSubClass", bound="MovementVector")
