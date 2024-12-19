import math
import os.path

import numpy as np
import pandas as pd
from natsort import natsorted
from skrf import Frequency, Network
from skrf.io.touchstone import Touchstone
from datetime import datetime, timedelta
from VNA_enums import DateFormats
from VNA_utils import open_full_results_df


class TouchstoneConverter:

    def __init__(self, touchstone_folder_path):
        # structure should be:
        # experiment name
        # -- experiment date + name
        # ---- gesture name
        # ------ touchstone files
        self.touchstone_folder_path = touchstone_folder_path
        self.output_data_frame: pd.DataFrame = None

    def get_experiment_names(self):
        return os.listdir(self.touchstone_folder_path)


class ExperimentTouchstones:

    def __init__(self, experiment_folder_path, experiment_name):
        self.experiment_folder_path = experiment_folder_path
        self.experiment_name: str = experiment_name
        self.exeriment_dates: [datetime] = self.get_dates_from_folder_names()

    def get_experiment_name(self):


    def get_dates_from_folder_names(self):
        directory = os.listdir(self.experiment_folder_path)
        return [
            datetime.strptime(item.split("_")[0], "%y%m%d%H%M") for item in os.listdir()
        ]


def get_time_recorded_from_touchstone(path):
    with open(path, "r") as file:
        lines = file.readlines()
        if len(lines) >= 2:
            line = lines[1].strip()
            time_recorded = line.split(" ", maxsplit=1)[1]
            return datetime.fromisoformat(time_recorded)
        else:
            pass


# class SParameter():
#     def __init__(self, sparam_string: str):
#


def s_params_as_list(network: Network, s_param, phase_flag=False, mag_flag=False):
    destination_port = int(s_param[1])
    origin_port = int(s_param[2])
    assert (1 <= destination_port <= 4) & (1 <= origin_port <= 4)
    if phase_flag:
        return [
            np.angle(matrix[destination_port - 1][origin_port - 1])
            for matrix in network.s
        ]
    elif mag_flag:
        return [
            abs(matrix[destination_port - 1][origin_port - 1]) for matrix in network.s
        ]
    else:
        return [matrix[destination_port - 1][origin_port - 1] for matrix in network.s]


def get_complex_phase_mag_lists(network: Network, s_param):
    complex_values = s_params_as_list(network, s_param)
    phase_values = s_params_as_list(network, s_param, phase_flag=True)
    db_values = s_params_as_list(network, s_param, mag_flag=True)
    return complex_values, phase_values, db_values


def space_out_duplicates(datetimes):
    spaced_out = []  # This will hold the adjusted datetimes
    i = 0

    while i < len(datetimes):
        current_datetime = datetimes[i]
        spaced_out.append(current_datetime)

        # Find the next different datetime (to detect consecutive duplicates)
        duplicates = [current_datetime]
        while i + 1 < len(datetimes) and datetimes[i + 1] == current_datetime:
            duplicates.append(datetimes[i + 1])
            i += 1

        # If duplicates were found, space them out by milliseconds
        if len(duplicates) > 1:
            num_duplicates = len(duplicates)
            ms_spacing = 1000 // num_duplicates  # Divide 1000ms evenly among duplicates

            for j in range(1, num_duplicates):
                new_datetime = current_datetime + timedelta(milliseconds=ms_spacing * j)
                spaced_out.append(new_datetime)

        i += 1

    return spaced_out


def zero_ref_time_column(df):
    df["time"] = pd.to_datetime(df["time"], format="%Y_%m_%d_%H_%M_%S.%f")
    df["time"] = df.groupby("id")["time"].transform(lambda x: x - x.min())
    df["time"] = df["time"].transform(lambda x: x.total_seconds())
    return df


def extract_data_from_touchstone_folder(folder_path, label) -> pd.DataFrame:
    df = None
    touchstone_files = natsorted(os.listdir(folder_path))[:3]
    times = [
        get_time_recorded_from_touchstone(os.path.join(folder_path, path))
        for path in touchstone_files
    ]
    times = space_out_duplicates(times)
    i = 0
    for touchstone_file in touchstone_files:
        path = os.path.join(folder_path, touchstone_file)
        print(f"{i} of {len(touchstone_files)}")
        df, times = extract_values_from_touchstone_files_to_df(path, label, times, df)
        i += 1
    return df


def create_empty_data_frame(network: Network) -> pd.DataFrame:
    frequency_array = network.f
    columns = ["id", "label", "mag_or_phase", "s_parameter", "time"] + [
        str(int(fq)) for fq in frequency_array
    ]
    return pd.DataFrame(columns=columns)


def linear_complex_value_to_dB(complex_value):
    return 20 * np.log10(np.abs(complex_value))


def extract_values_from_touchstone_files_to_df(
    path, experiment_label, times, df: pd.DataFrame = None
) -> pd.DataFrame:
    touchstone_network = Network(path)
    touchstone_class = Touchstone(path)
    if df is None:
        df = create_empty_data_frame(touchstone_network)

    fname = os.path.basename(path)
    split_fname = fname.split("_")

    experiment_id = ("_").join(split_fname[:6])

    time_touchstone_created = times.pop(0)

    gesture_label = split_fname[7]
    label = experiment_label + "_" + gesture_label

    s_params = [f"S{i}{j}" for i in range(1, 5) for j in range(1, 5)]

    for s_param in s_params:
        complex_values, phase_values, db_values = get_complex_phase_mag_lists(
            touchstone_network, s_param
        )
        print(
            f"{s_param} time {time_touchstone_created.strftime(DateFormats.MILLISECONDS.value)}"
        )
        phase_row = [
            experiment_id,
            label,
            "phase",
            s_param,
            time_touchstone_created.strftime(DateFormats.MILLISECONDS.value),
        ] + phase_values
        db_row = [
            experiment_id,
            label,
            "magnitude",
            s_param,
            time_touchstone_created.strftime(DateFormats.MILLISECONDS.value),
        ] + db_values

        df.loc[len(df)] = phase_row
        df.loc[len(df)] = db_row
    return df, times


if __name__ == "__main__":
    # iterate through whole folder
    path = r"D:\Nik\phantom\moving"
    # label is the same for each
    label = "rotating_test"
    df = extract_data_from_touchstone_folder(path, label)

    full_df = open_full_results_df(
        r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\full_dfs\full_combined_df_2024_08_09.pkl"
    )
