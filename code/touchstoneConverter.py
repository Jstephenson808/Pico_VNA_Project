import math
import os.path
import re

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
        self.experiment_folders: [ExperimentFolder] = self.open_experiment_folders()

    def get_experiment_names(self):
        return os.listdir(self.touchstone_folder_path)

    def open_experiment_folders(self):
        experiment_names = self.get_experiment_names()

        return [
            ExperimentFolder(
                os.path.join(self.touchstone_folder_path, experiment_name),
                experiment_name,
            )
            for experiment_name in experiment_names
        ]

    def extract_all_touchstone_data_to_dataframe(self):
        for experiment_folder in self.experiment_folders:
            experiment_data_frame = experiment_folder.extract_data_for_each_experiment()
            self.output_data_frame = pd.concat(
                [self.output_data_frame, experiment_data_frame]
            )


class ExperimentFolder:
    """
    This class contains the folder for a given experiment label, within this folder is the date labeled
    experiment folders
    """

    def __init__(self, experiment_folder_path, experiment_name):
        self.experiment_folder_path = experiment_folder_path
        self.experiment_name: str = experiment_name
        self.experiment_folder = os.listdir(self.experiment_folder_path)
        self.experiment_dates: [datetime] = self.get_dates_from_folder_names()
        self.individual_experiments: [IndividualExperiment] = (
            self.open_individual_experiment_folder()
        )

    def get_dates_from_folder_names(self):
        return [
            datetime.strptime(item.split("_")[0], "%y%m%d%H%M")
            for item in self.experiment_folder
        ]

    def open_individual_experiment_folder(self):
        return [
            IndividualExperiment(folder, self.experiment_name)
            for folder in self.experiment_folder
        ]

    def extract_data_for_each_experiment(self):
        experiment_data_frame = None
        for experiment in self.individual_experiments:
            individual_experiment_data_frame = (
                experiment.extract_data_for_individual_experiment()
            )
            experiment_data_frame = pd.concat(
                [experiment_data_frame, individual_experiment_data_frame]
            )
        return experiment_data_frame


class IndividualExperiment:
    """
    This folder contains all the experiment repeats related to the one experiment title, these are timestamped
    """

    def __init__(self, individual_experiment_folder, experiment_name):
        self.individual_experiment_folder = individual_experiment_folder
        self.experiment_name = experiment_name
        self.experiment_timestamp = self.get_experiment_timestamp()
        self.gestures_folders: [GesturesFolder] = self.open_gestures_folder()

    def get_experiment_timestamp(self):
        return datetime.strptime(
            os.path.basename(self.individual_experiment_folder).split("_")[0],
            "%y%m%d%H%M",
        )

    def get_gesture_folder_list(self):
        return os.listdir(self.individual_experiment_folder)

    def open_gestures_folder(self):
        return [
            GesturesFolder(
                os.path.join(self.individual_experiment_folder, path),
                self.experiment_name,
                self.experiment_timestamp,
            )
            for path in self.get_gesture_folder_list()
        ]

    def extract_data_for_individual_experiment(self):
        full_experiment_data_frame = None
        for gesture_folder in self.gestures_folders:
            individual_gesture_data_frame = (
                gesture_folder.extract_data_for_individual_gesture()
            )
            full_experiment_data_frame = pd.concat(
                [full_experiment_data_frame, individual_gesture_data_frame]
            )
        return full_experiment_data_frame


class GesturesFolder:
    """
    This folder contains folders with all the captured gestures for a given individual experiment.
    """

    def __init__(self, gesture_folder_path, experiment_name, timestamp):
        self.gestures_folder_path = gesture_folder_path
        self.experiment_name = experiment_name
        self.timestamp = timestamp
        self.folder_list = self.set_folder_list()
        self.gesture_label_list = self.get_gesture_list()
        self.gesture_folders: [GesturesFolder] = self.open_gesture_folder()
        self.data_frame = None

    def set_folder_list(self):
        return os.listdir(self.gestures_folder_path)

    def get_gesture_list(self):
        pattern = r"Gesture_(.*)"
        compiled_pattern = re.compile(pattern)
        gesture_labels = [
            re.search(compiled_pattern, gesture).group(1)
            for gesture in self.folder_list
        ]
        return gesture_labels

    def open_gesture_folder(self):
        return [
            GestureFolder(
                os.path.join(self.gestures_folder_path, gesture_folder),
                self.experiment_name,
                gesture_label,
                self.timestamp,
            )
            for gesture_folder, gesture_label in zip(
                self.folder_list, self.gesture_label_list
            )
        ]

    def extract_data_for_individual_gesture(self) -> pd.DataFrame:
        data_frame = None
        for gesture_folder in self.gesture_folders:
            new_data_frame = gesture_folder.extract_data_for_each_gesture_repeat()
            data_frame = pd.concat([data_frame, new_data_frame])
        return data_frame


class GestureFolder:
    """
    This folder contains all the repeats for a given gesture for a
    specific experiment the folder names contain the number of the repeat
    in the format 1 -> n_repeats
    """

    def __init__(self, gesture_path, experiment_name, gesture_label, timestamp):
        self.gesture_path = gesture_path
        self.experiment_name = experiment_name
        self.gesture_label = gesture_label
        self.timestamp = timestamp
        self.individual_gesture_tests: [IndividualGestureCapture] = (
            self.open_individual_gesture_test()
        )
        self.n_repeats_of_gesture = len(self.individual_gesture_tests)

    def open_individual_gesture_test(self):
        return [
            IndividualGestureCapture(
                os.path.join(self.gesture_path, folder),
                self.experiment_name,
                self.gesture_label,
                self.timestamp,
                folder,
            )
            for folder in os.listdir(self.gesture_path)
        ]

    def extract_data_for_each_gesture_repeat(self) -> pd.DataFrame:
        data_frame = None
        for gesture_test in self.individual_gesture_tests:
            new_data_frame = gesture_test.extract_data_from_individual_gesture_capture()
            data_frame = pd.concat([data_frame, new_data_frame])
        return data_frame


class IndividualGestureCapture:
    def __init__(
        self,
        individual_gesture_folder_path,
        experiment_name,
        gesture_label,
        timestamp,
        repeat_number,
    ):
        self.individual_gesture_folder_path = individual_gesture_folder_path
        self.experiment_name = experiment_name
        self.gesture_label = gesture_label
        self.timestamp: datetime = timestamp
        self.repeat_number = repeat_number
        self.touchstone_files: [TouchstoneFile] = self.open_touchstones()
        self.data_frame = self.create_empty_data_frame()

    def sort_folder(self):
        return sorted(
            os.listdir(self.individual_gesture_folder_path),
            key=lambda x: int(x.split("_")[1]),
        )

    def get_path_to_touchstone(self, touchstone_fname):
        return os.path.join(self.individual_gesture_folder_path, touchstone_fname)

    def open_touchstones(self):
        sorted_folder = self.sort_folder()
        return [
            TouchstoneFile(
                self.get_path_to_touchstone(fname),
                self.experiment_name,
                self.timestamp,
                self.gesture_label,
                self.timestamp,
                self.repeat_number,
            )
            for fname in sorted_folder
        ]

    def extract_data_from_individual_gesture_capture(self) -> pd.DataFrame:

        ##### atomic section #####

        # need to sort touchstone files by number
        self.touchstone_files = sorted(self.touchstone_files)
        # get list of all the recording times to space them out correctly
        times = [
            touchstone.touchstone_time_recorded for touchstone in self.touchstone_files
        ]
        times = self.space_out_touchstone_recording_times(times)
        zero_referenced_times = self.zero_ref_recording_time(times)
        # as the files were sorted at the start this is
        for time, zero_referenced_time, touchstone in zip(
            times, zero_referenced_times, self.touchstone_files
        ):
            touchstone.touchstone_time_recorded = time
            touchstone.zero_referenced_time_recorded = zero_referenced_time

        ##### end atomic section #####

        i = 0
        print(
            f"Experiment {self.experiment_name} Timestamp {self.timestamp.strftime('%d/%m/%Y, %H:%M:%S')} Gesture {self.gesture_label}"
        )
        for touchstone_file in self.touchstone_files:
            print(f"{i} of {len(self.touchstone_files)}")
            self.data_frame, times = (
                touchstone_file.extract_values_from_touchstone_files_to_df(
                    self.data_frame, times
                )
            )
            i += 1
        return df

    def zero_ref_recording_time(self, datetimes: [datetime]):
        start_time: datetime = datetimes[0]
        zero_referenced_times = [time - start_time for time in datetimes]
        return zero_referenced_times

    def space_out_touchstone_recording_times(self, datetimes: [datetime]):
        """
        Times for the touchstone recording are only accurate to 1second so if more than one recording
        is done per second then these need to be spaced out, this is done by evenly spacing out files for each second
        Args:
            datetimes:

        Returns:

        """
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
                ms_spacing = (
                    1000 // num_duplicates
                )  # Divide 1000ms evenly among duplicates

                for j in range(1, num_duplicates):
                    new_datetime = current_datetime + timedelta(
                        milliseconds=ms_spacing * j
                    )
                    spaced_out.append(new_datetime)

            i += 1

        return spaced_out

    def create_empty_data_frame(self) -> pd.DataFrame:
        frequency_array = self.touchstone_files[0].touchstone_network.f
        columns = ["id", "label", "mag_or_phase", "s_parameter", "time"] + [
            str(int(fq)) for fq in frequency_array
        ]
        return pd.DataFrame(columns=columns)


class TouchstoneFile:
    def __init__(
        self, touchstone_path, experiment_name, gesture_label, timestamp, repeat_number
    ):
        self.touchstone_path = touchstone_path
        self.touchstone_network = Network(self.touchstone_path)
        self.experiment_name = experiment_name
        self.gesture_label = gesture_label
        self.experiment_timestamp: datetime = timestamp
        self.repeat_number = repeat_number
        self.touchstone_number = self.get_touchstone_number()
        self.touchstone_time_recorded: datetime = (
            self.get_time_recorded_from_touchstone()
        )
        self.zero_referenced_time: timedelta = None

    def __eq__(self, other):
        return (
            self.touchstone_number == other.touchstone_number
            and self.experiment_name == other.experiment_name
            and self.gesture_label == other.gesture_label
        )

    def __lt__(self, other):
        return self.touchstone_number < other.touchstone_number

    def __gt__(self, other):
        return self.touchstone_number > other.touchstone_number

    def get_touchstone_number(self):
        fname = os.path.basename(self.touchstone_path)
        number = fname.split("_")[1].split(".")[0]
        return int(number)

    def get_time_recorded_from_touchstone(self):
        with open(self.touchstone_path, "r") as file:
            lines = file.readlines()
            if len(lines) >= 2:
                line = lines[1].strip()
                time_recorded = line.split(" ", maxsplit=1)[1]
                return datetime.fromisoformat(time_recorded)
            else:
                pass

    def create_experiment_id(self):
        return f"{self.experiment_name}_{self.experiment_timestamp.strftime(DateFormats.VNA_FOLDER_DATE_FROMAT.value)}"

    def extract_values_from_touchstone_files_to_df(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:

        # id for the experiment as a whole
        experiment_id = self.create_experiment_id()

        # id for the gesture
        label = f"{self.experiment_name}_{self.gesture_label}"

        s_params = [f"S{i}{j}" for i in range(1, 5) for j in range(1, 5)]

        for s_param in s_params:
            complex_values, phase_values, db_values = self.get_complex_phase_mag_lists(
                s_param
            )

            phase_row = [
                experiment_id,
                label,
                "phase",
                s_param,
                self.zero_referenced_time.total_seconds(),
            ] + phase_values

            db_row = [
                experiment_id,
                label,
                "magnitude",
                s_param,
                self.zero_referenced_time.total_seconds(),
            ] + db_values

            df.loc[len(df)] = phase_row
            df.loc[len(df)] = db_row
        return df

    def get_complex_phase_mag_lists(self, s_param):
        origin_port, destination_port = self.get_port_numbers(s_param)
        complex_values = self.get_complex_values_as_list(origin_port, destination_port)
        phase_values = self.get_phase_as_list(origin_port, destination_port)
        db_values = self.get_magnitute_as_list(origin_port, destination_port)
        return complex_values, phase_values, db_values

    def get_port_numbers(self, s_param):
        destination_port = int(s_param[1])
        origin_port = int(s_param[2])
        assert (1 <= destination_port <= 4) & (1 <= origin_port <= 4)
        return destination_port, origin_port

    def get_phase_as_list(self, origin_port, destination_port):
        return [
            np.angle(matrix[destination_port - 1][origin_port - 1])
            for matrix in self.touchstone_network.s
        ]

    def get_magnitute_as_list(self, origin_port, destination_port):
        return [
            abs(matrix[destination_port - 1][origin_port - 1])
            for matrix in self.touchstone_network.s
        ]

    def get_complex_values_as_list(self, origin_port, destination_port):
        return [
            matrix[destination_port - 1][origin_port - 1]
            for matrix in self.touchstone_network.s
        ]


def zero_ref_time_column(df):
    df["time"] = pd.to_datetime(df["time"], format=DateFormats.MILLISECONDS.value)
    df["time"] = df.groupby("id")["time"].transform(lambda x: x - x.min())
    df["time"] = df["time"].transform(lambda x: x.total_seconds())
    return df


if __name__ == "__main__":
    # iterate through whole folder
    path = r"D:\Nik\phantom\moving"
    # label is the same for each
    label = "rotating_test"
    df = extract_data_from_touchstone_folder(path, label)

    full_df = open_full_results_df(
        r"C:\Users\js637s.CAMPUS\PycharmProjects\Pico_VNA_Project\pickles\full_dfs\full_combined_df_2024_08_09.pkl"
    )
