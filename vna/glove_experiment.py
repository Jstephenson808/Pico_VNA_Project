import pandas as pd

from VNA_utils import open_pickled_object
from vna.VNA_enums import SParam, MagnitudeOrPhase
from vna.VNA_utils import (
    open_pickled_object_in_pickle_folder,
    hz_to_mhz,
    mhz_to_hz,
    retype_str_fq_columns_to_int,
)
from vna.graphs import plot_fq_time_series
from vna.ml_model import fix_measurement_column

full_data_frame: pd.DataFrame = open_pickled_object_in_pickle_folder(
    "glove_experiment_results.pkl"
)
retype_str_fq_columns_to_int(full_data_frame)
gesture_repeated = full_data_frame.query(
    "id == 'liquid_metal_glove_6ges_same_gesture_10time_2412181543'"
)
plot_fq_time_series(
    gesture_repeated,
    s_parameter=SParam.S21,
    mag_or_phase=MagnitudeOrPhase.Magnitude,
    label="liquid_metal_glove_6ges_same_gesture_10time_1",
    n_random_ids=1,
    target_frequency=mhz_to_hz(200),
)
# full_data_frame.query("")
