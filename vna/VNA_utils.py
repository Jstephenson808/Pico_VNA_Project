import os
import pickle
from pathlib import Path
from random import choice
from time import time, sleep
from typing import Type, TypeVar

import re


import numpy as np
import pandas as pd

import VNA_exceptions
import VNA_defaults
from vna.VNA_enums import (
    ClassificationResultsColumns,
    ClassificationResultsAccuracy,
    MagnitudeOrPhase,
    DataFrameCols,
)
from vna.scipiCommands import SParam

T = TypeVar("T")


def countdown_timer(seconds):
    while seconds > 0:
        print(f"{seconds}..")
        sleep(1)
        seconds -= 1
    print("Start")


# todo return execution time
def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        execution_time = t2 - t1
        print(f"Function {func.__name__!r} executed in {execution_time:.4f}s")
        return result

    return wrap_func


def mhz_to_hz(mhz):
    """
    utility function to convert mhz to hz
    :param mhz: MHz value
    :return: value in Hz
    """
    return mhz * 1_000_000


def hz_to_mhz(hz):
    """
    utility function to convert hz to Mhz
    :param hz: Hz value
    :return: value in MHz
    """
    return hz / 1_000_000


def ghz_to_hz(ghz):
    """
    utility function to convert GHz to Hz
    :param ghz: GHz value
    :return: value in Hz
    """
    return ghz * 1_000_000_000


def hz_to_ghz(hz):
    """
    utility function to convert Hz to GHz
    :param hz: Hz value
    :return: value in GHz
    """
    return hz / 1_000_000_000


def get_root_folder_path():
    """
    utility function to get the root folder of the project
    assumes file is running in original folder, this can cause issues
    if the root folder is moved
    :return: path to the root folder
    """
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(path) != VNA_defaults.ROOT_FOLDER:
        raise VNA_exceptions.FileNotInCorrectFolder(
            f"Code (.py files) isn't in correct folder for this function to work,"
            f" move into a folder below root dir, currently in {__file__}"
        )
    return path


def get_results_path() -> str:
    path = os.path.join(get_root_folder_path(), VNA_defaults.RESULTS_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_graph_path():
    path = os.path.join(get_results_path(), VNA_defaults.GRAPH_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_data_path() -> str:
    path = os.path.join(get_results_path(), VNA_defaults.DATA_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_pickle_path() -> str:
    path = os.path.join(get_root_folder_path(), VNA_defaults.PICKLE_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_full_df_path() -> str:
    path = os.path.join(get_pickle_path(), VNA_defaults.COMBINED_DF_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_classifier_path():
    path = os.path.join(get_pickle_path(), VNA_defaults.CLASSIFIER_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_calibration_path():
    path = os.path.join(get_root_folder_path(), VNA_defaults.CALIBRATION_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_classifiers_path():
    return os.path.join(get_pickle_path(), "classifiers")


def get_full_dfs_path():
    return os.path.join(get_pickle_path(), "full_dfs")


def get_frequency_column_headings_list(df: pd.DataFrame) -> [int]:
    return [int(x) for x in df.columns[5:]]


def get_full_results_df_path():
    return os.path.join(get_pickle_path(), "full_results_dfs")


def get_touchstones_path():
    return os.path.join(get_results_path(), VNA_defaults.TOUCHSTONES_FOLDER)


def reorder_data_frame_columns(
    df: pd.DataFrame, new_order_indexes: [int]
) -> pd.DataFrame:
    columns = list(df.columns)
    new_columns = sorted(
        columns, key=lambda x: new_order_indexes.index(columns.index(x))
    )
    return df[new_columns]


def input_movement_label() -> str:
    label = input("Provide gesture label or leave blank for none:")
    return label


def pickle_object(object_to_pickle, *, path: str, file_name: str):
    os.makedirs(path, exist_ok=True)
    if ".pkl" not in file_name[-4:]:
        file_name = f"{file_name}.pkl"
    path = os.path.join(path, file_name)
    with open(path, "wb") as f:
        pickle.dump(object_to_pickle, f)


def load_pickled_object_as_type(path: Path, expected_type: Type[T]) -> T:
    if not path.exists():
        raise FileNotFoundError(f"The path {path} does not exist")

    try:
        unpickled_object = open_pickled_object(path)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Could not load a pickle file from {path}: {e}")
    if not isinstance(unpickled_object, expected_type):
        raise TypeError(f"Expected {expected_type} but got {type(unpickled_object)}")
    return unpickled_object


def open_pickled_object(path):
    with open(path, "rb") as f:
        unpickled = pickle.load(f)
    return unpickled


def open_full_results_df(file_name, folder=None) -> pd.DataFrame:
    """
    Opens a .pkl data frame within the folder provided, if folder arg is none
    then the default folder is used
    :param file_name: the file name of the target data frame
    :param folder: the folder of the data frame
    :return: data frame
    """
    if folder is None:
        folder = get_full_df_path()

    return open_pickled_object(os.path.join(folder, file_name))


def get_label_from_pkl_path(path: Path):
    """
    removes .pkl and then date from fname format
    "all_Sparams_magnitude_0.01_0.11_2024_04_02.pkl"
    """
    return os.path.basename(path)[::-1].split("_", maxsplit=3)[-1][::-1]


def linear_complex_value_to_dB(complex_value):
    return 20 * np.log10(np.abs(complex_value))


def convert_magnitude_to_db(magnitude_value: float):
    return 20 * np.log10(magnitude_value)


def convert_magnitude_to_db_array(values: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(values, 1e-12))  # avoid log(0)


def convert_magnitude_rows_to_db(data_frame: pd.DataFrame):
    mask = data_frame["mag_or_phase"] == "magnitude"
    cols = data_frame.columns[5:]

    data_frame.loc[mask, cols] = convert_magnitude_to_db_array(
        data_frame.loc[mask, cols].to_numpy()
    )

    return data_frame


def extract_captured_gestures_from_results_df(results_df: pd.DataFrame) -> list:
    return [
        gesture
        for gesture in list(results_df[ClassificationResultsColumns.GESTURE].unique())
        if gesture not in list(ClassificationResultsAccuracy)
    ]


def format_enum_list(items):
    enum_list = [
        f"{re.sub(r'[^A-Z0-9]+', '_', item.upper()).strip('_')} = '{item}'"
        for item in items
    ]
    return enum_list


def get_list_of_s_params_in_df(df: pd.DataFrame) -> [SParam]:
    return [SParam[sparam_string] for sparam_string in df["s_parameter"].unique()]


def filter_between_frequency(df, low_frequency, high_frequency):
    columns_to_drop = list(
        filter(lambda x: (low_frequency > x) | (x > high_frequency), df.columns[5:])
    )
    return df.drop(columns_to_drop, axis=1)


def extract_random_single_gesture_for_each_experiment_to_df(
    capture_df: pd.DataFrame, target_s_param: SParam, mag_or_phase: MagnitudeOrPhase
) -> pd.DataFrame:
    experiments = capture_df[DataFrameCols.LABEL.value].unique()
    output_df = None
    for experiment in experiments:
        # get all the same label experiments -> this means the same gesture
        same_gesture = capture_df[
            (capture_df[DataFrameCols.LABEL.value] == experiment)
            & (capture_df[DataFrameCols.S_PARAMETER.value] == target_s_param.value)
            & (capture_df["mag_or_phase"] == mag_or_phase.value)
        ]

        single_gesture = same_gesture[
            same_gesture["id"] == choice(same_gesture["id"].unique())
        ]
        # output will contain one unique gesture capture for each
        output_df = pd.concat([output_df, single_gesture], ignore_index=True)
    return output_df


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_cols = df.columns[df.columns.duplicated()].unique()
    new_cols = {}

    for col in duplicate_cols:
        cols_with_name = df.loc[:, df.columns == col]
        combined = cols_with_name.bfill(axis=1).iloc[:, 0]
        new_cols[col] = combined

    # Drop all duplicates at once
    df = df.drop(columns=[col for col in df.columns if col in duplicate_cols])

    # Combine all at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def list_all_pkl_files_from_folder(folder_path: Path) -> [Path]:
    return list(folder_path.glob("**/*.pkl"))
