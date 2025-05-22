import pandas as pd
import numpy as np
from vna.VNA_utils import coalesce_duplicate_columns


def test_coalesce_duplicate_columns():
    df = pd.DataFrame(
        np.array([[1, np.nan], [2, np.nan], [np.nan, 3], [np.nan, 4]]),
        columns=["a", "a"],
    )

    expected = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})

    result = coalesce_duplicate_columns(df)

    pd.testing.assert_frame_equal(result, expected)
