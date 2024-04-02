import os
from time import time
import VNA_exceptions
import VNA_defaults

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
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