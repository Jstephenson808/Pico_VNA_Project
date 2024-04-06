import os
from time import time, sleep

import pandas as pd

import VNA_exceptions
import VNA_defaults


def countdown_timer(seconds):
    while seconds > 0:
        print(f"{seconds}..")
        sleep(1)
        seconds -= 1
    print("Start")

#todo return execution time
def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        execution_time = t2-t1
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

def get_results_path():
    path = os.path.join(get_root_folder_path(), VNA_defaults.RESULTS_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_data_path():
    path = os.path.join(get_results_path(), VNA_defaults.DATA_FOLDER)
    os.makedirs(path, exist_ok=True)
    return path


def get_pickle_path():
    path = os.path.join(get_root_folder_path(), VNA_defaults.PICKLE_FOLDER)
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

def get_frequency_column_headings_list(df: pd.DataFrame)->[int]:
    return [x for x in df.columns if isinstance(x, int)]

def get_full_results_df_path():
    return os.path.join(get_pickle_path(), "full_results_dfs")

def reorder_data_frame_columns(df:pd.DataFrame, new_order_indexes:[int])->pd.DataFrame:
    columns = list(df.columns)
    new_columns = sorted(columns, key=lambda x: new_order_indexes.index(columns.index(x)))
    return df[new_columns]
