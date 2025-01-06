import os
import pickle
from time import time, sleep

import numpy as np
import pandas as pd

import VNA_exceptions
import VNA_defaults


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


def get_classifier_path() -> str:
    path = os.path.join(get_pickle_path(), VNA_defaults.CLASSIFIER_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_calibration_path() -> str:
    path = os.path.join(get_root_folder_path(), VNA_defaults.CALIBRATION_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_classifiers_path() -> str:
    return os.path.join(get_pickle_path(), "classifiers")


def get_full_dfs_path() -> str:
    return os.path.join(get_pickle_path(), "full_dfs")


def get_frequency_column_headings_list(df: pd.DataFrame) -> [int]:
    return [int(x) for x in df.columns[5:]]


def retype_str_fq_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    new_fq_col_headings = [
        int(col_heading) for col_heading in get_frequency_column_headings_list(df)
    ]
    new_col_headings = get_none_fq_columns(df) + new_fq_col_headings
    df.columns = new_col_headings
    return df


def get_none_fq_columns(df: pd.DataFrame):
    return list(df.columns[:5])


def get_full_results_df_path() -> str:
    return os.path.join(get_pickle_path(), "full_results_dfs")


def get_touchstones_path() -> str:
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


def pickle_object(
    object_to_pickle, *, folder_path: str = get_pickle_path(), file_name: str
):
    os.makedirs(folder_path, exist_ok=True)
    if ".pkl" not in file_name[-4:]:
        file_name = f"{file_name}.pkl"
    folder_path = os.path.join(folder_path, file_name)
    with open(folder_path, "wb") as f:
        pickle.dump(object_to_pickle, f)


def open_pickled_object_in_pickle_folder(file_name: str):
    return open_pickled_object(os.path.join(get_pickle_path(), file_name))


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


def get_label_from_pkl_path(path):
    """
    removes .pkl and then date from fname format
    "all_Sparams_magnitude_0.01_0.11_2024_04_02.pkl"
    """
    return os.path.basename(path)[::-1].split("_", maxsplit=3)[-1][::-1]


def convert_magnitude_to_db(magnitude_value: float):
    return 20 * np.log10(magnitude_value)


def convert_magnitude_rows_to_db(data_frame: pd.DataFrame):
    magnitude = data_frame.query("mag_or_phase == 'magnitude'")
    frequency_values: pd.DataFrame = magnitude.iloc[:, 5:]
    frequency_values.apply(convert_magnitude_to_db, axis=1)
