from __future__ import annotations

import pandas as pd

from vna.s_parameter_data import SParameterData


class SParameterDataConverter:
    """
    A class to handle conversions of SParameterData objects to different formats.
    """

    @staticmethod
    def convert_to_long_format(s_param_data: SParameterData) -> SParameterData:
        """
        Converts the DataFrame within an SParameterData object from a "wide" format
        (where frequencies are columns) to a "long" or "tidy" format.

        The resulting tidy format has the following columns:
        - id
        - time
        - label
        - s_parameter
        - frequency
        - magnitude
        - phase

        Args:
            s_param_data: An SParameterData object with data in the wide format.

        Returns:
            A new SParameterData object containing the data in the long format.
        """
        wide_df = s_param_data.data_frame

        # Identify the columns that are not frequencies (the "ID" columns)
        id_vars = [
            col
            for col in wide_df.columns
            if not isinstance(col, (int, float))
        ]

        # Identify the frequency columns that need to be unpivoted
        value_vars = s_param_data.get_frequency_columns()

        # Use pd.melt() to unpivot the frequency columns
        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="frequency",
            value_name="value",
        )

        # Use pivot_table to separate magnitude and phase into their own columns
        # This is the key step to create the final tidy structure.
        tidy_df = long_df.pivot_table(
            index=[
                "id",
                "time",
                "label",
                "s_parameter",
                "frequency",
            ],
            columns="mag_or_phase",
            values="value",
        ).reset_index()

        # Clean up the column names after the pivot
        tidy_df.columns.name = None
        tidy_df = tidy_df.rename(
            columns={"magnitude": "magnitude", "phase": "phase"}
        )

        # Create a new SParameterData object with the transformed data
        new_label = f"{s_param_data.label}_tidy"
        return SParameterData(label=new_label, data_frame=tidy_df)
