from VNA_calibration import VnaCalibration
from VNA_data import VnaData

import os
import win32com.client

from datetime import datetime, timedelta
import pandas as pd

from VNA_enums import (
    MeasurementFormat,
    SParam,
    MeasureSParam,
    DateFormats,
)
from VNA_exceptions import *
from VNA_utils import (
    get_data_path,
    get_root_folder_path,
    countdown_timer
)


class VNA:

    @staticmethod
    def file_label_input() -> str:
        file_label = input(
            "Input label for file (no spaces) or press enter for no label:"
        )
        while (file_label != "") and not VnaData.test_file_name(file_label):
            file_label = input("Incorrect format try again or enter to skip:")
        return file_label

    def __init__(
        self,
        calibration: VnaCalibration,
        vna_data: VnaData,
        vna_string="PicoControl2.PicoVNA_2",
    ):
        self.calibration = calibration
        self.vna_object = win32com.client.gencache.EnsureDispatch(vna_string)
        self.output_data = vna_data

    #todo close connection
    def connect(self):
        print("Connecting VNA")
        search_vna = self.vna_object.FND()
        if search_vna == 0:
            raise VNAError("Connection Failed, do you have Pico VNA Open?")
        print(f"VNA {str(search_vna)} Loaded")

    def load_cal(self):
        """
        loads the calibration which is stored in object
        :return:
        """
        print("Loading Calibration")
        ans = self.vna_object.LoadCal(self.calibration.calibration_path)
        if ans != "OK":
            raise VNAError(f"Calibration Failure {ans}")
        print(f"Result {ans}")

    def get_data(
        self, s_parameter: SParam, data_format: MeasurementFormat, point=0
    ) -> str:
        """
        wrapper for getting data from the VNA after measurement
        :param s_parameter: S Param data to be returned
        :param data_format: measurement requested
        :param point:
        :return: data string which is ',' separted in the format "freq, measurement_value_at_freq, freq, measurement_value_at_freq,..."
        """
        return self.vna_object.GetData(s_parameter.value, data_format.value, point)

    def split_data_string(self, data_string: str):
        """

        :param data_string: Data string which is ',' separted in the format "freq, measurement_value_at_freq, freq,
                            measurement_value_at_freq" output from self.get_data()
        :return: tuple containing list of frequencies and list of data values
        """
        data_list: list[str] = data_string.split(",")
        frequencies = data_list[::2]
        data = data_list[1::2]
        return frequencies, data

    def vna_data_string_to_df(
        self,
        elapsed_time: timedelta,
        magnitude_data_string: str,
        phase_data_string: str,
        s_parameter: SParam,
        label: str,
        id,
    ) -> pd.DataFrame:
        """
        Converts the strings returned by the VNA .get_data method into a data frame
        with the elapsed time, measured SParam, frequency, mag and phase
        :param elapsed_time: timedelta representing elapsed time when the reading was taken
        :param magnitude_data_string: data string returned by get_data method with magnitude argument
        :param phase_data_string: phase data string returned by get_data method with phase argument
        :param s_parameter: SParam enum value represting the measured Sparam
        :return: pd dataframe formatted correctly to be appended to the data frame in memory
        """
        # todo fix this so you can have phase or mag independently
        frequencies, magnitudes = self.split_data_string(magnitude_data_string)
        frequencies, phases = self.split_data_string(phase_data_string)
        time_float = float(f"{elapsed_time.seconds}.{elapsed_time.microseconds}")
        data_dict = {
            DataFrameCols.ID.value: id,
            DataFrameCols.TIME.value: [time_float for _ in frequencies],
            DataFrameCols.LABEL.value: [label for _ in frequencies],
            DataFrameCols.S_PARAMETER.value: [s_parameter for _ in frequencies],
            DataFrameCols.FREQUENCY.value: [int(fq) for fq in frequencies],
            DataFrameCols.MAGNITUDE.value: [float(mag) for mag in magnitudes],
            DataFrameCols.PHASE.value: [float(phase) for phase in phases],
        }
        return pd.DataFrame(data_dict)

    def generate_output_path(
        self,
        output_folder: str,
        s_params_saved: SParam,
        run_time: timedelta,
        fname="",
        label="",
    ):
        """
        Utility function to generate file name and join it to path
        :param s_params_measure: measured s parameteres
        :param run_time:
        :param fname:
        :return:
        """
        if fname != "" and label != "":
            label_fname = ("_").join((fname, label))
        else:
            label_fname = ("").join((fname, label))

        if label == "":
            label = datetime.now().strftime(DateFormats.DATE_FOLDER.value)

        if label_fname != "":
            label_fname += "_"

        s_params = ("_").join([s_param.value for s_param in s_params_saved])
        filename = f"{label_fname}{datetime.now().strftime(DateFormats.CURRENT.value)}_{s_params}_{run_time.seconds}_secs.csv"
        return os.path.join(get_root_folder_path(), output_folder, label, filename)

    # todo what if you only want to measure one of phase or logmag?
    def add_measurement_to_data_frame(
        self, s_param: SParam, elapsed_time: timedelta, label: str, id
    ):
        """
        Gets current measurement strings (logmag and phase) for the given S param from VNA and converts it
        to a pd data frame, appending this data frame to the output data
        :param s_param: SParam to get the data
        :param elapsed_time: The elaspsed time of the current test (ie the time the data was captured, referenced to 0s)
        :return: the data frame concated on to the current output
        """
        df = self.vna_data_string_to_df(
            elapsed_time,
            self.get_data(s_param, MeasurementFormat.LOGMAG),
            self.get_data(s_param, MeasurementFormat.PHASE),
            s_param.value,
            label,
            id,
        )
        return pd.concat([self.output_data.data_frame, df])

    # add in timer logging
    def measure_wrapper(self, str):
        return self.vna_object.Measure(str)

    #@timer_func
    def take_measurement(
        self,
        s_params_measure: MeasureSParam,
        s_params_output: [SParam],
        elapsed_time: timedelta,
        label: str,
        id,
    ):
        """
        Takes measurement on the VNA, processes it and appends it to the output_data.data_frame
        df
        :param s_params_measure: The S params for the VNA to measure, using
        :param s_params_output:
        :param elapsed_time:
        """

        self.measure_wrapper(s_params_measure.value)

        for s_param in s_params_output:
            self.output_data.data_frame = self.add_measurement_to_data_frame(
                s_param, elapsed_time, label, id
            )

    def input_movement_label(self) -> str:
        label = input("Provide gesture label or leave blank for none:")
        return label

    def measure(
        self,
        run_time: timedelta,
        s_params_measure: MeasureSParam = MeasureSParam.ALL,
        s_params_output: [SParam] = None,
        file_name: str = "",
        output_dir=get_data_path(),
        label=None,
        *,
        print_countdown=False,
    ) -> VnaData:

        # label = 'test'
        if label is None:
            label = self.input_movement_label()

        if s_params_output == None:
            s_params_output = [SParam.S11]


        self.connect()
        self.load_cal()

        self.output_data.csv_path = self.generate_output_path(output_dir, s_params_output, run_time, file_name, label)
        os.makedirs(os.path.dirname(self.output_data.csv_path), exist_ok=True)
        print(f"Saving to {self.output_data.csv_path}")
        countdown_timer(2)
        start_time = datetime.now()
        finish_time = start_time + run_time
        current_time = datetime.now()
        measurement_number = 0
        while current_time < finish_time:
            current_time = datetime.now()
            elapsed_time = current_time - start_time
            if print_countdown:
                print(f"Running for another {(run_time - elapsed_time)}")
            self.take_measurement(
                s_params_measure,
                s_params_output,
                elapsed_time,
                label,
                id=start_time.strftime(DateFormats.CURRENT.value),
            )
            measurement_number += 1
            if measurement_number % 10 == 0:

                self.output_data.data_frame.to_csv(
                    self.output_data.csv_path, index=False
                )

        self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.vna_object.CloseVNA()
        print("VNA Closed")
        return self.output_data

    def measure_n_times(
            self,
            run_time: timedelta,
            s_params_measure: MeasureSParam = MeasureSParam.ALL,
            s_params_output: [SParam] = None,
            file_name: str = "",
            output_dir=get_data_path(),
            label=None,
            *,
            print_countdown=False,
            n_measures=50
    ) -> VnaData:

        # label = 'test'
        if label is None:
            label = self.input_movement_label()

        if s_params_output == None:
            s_params_output = [SParam.S11]


        self.connect()
        self.load_cal()

        for i in range(n_measures):
            self.output_data.csv_path = self.generate_output_path(
                output_dir, s_params_output, run_time, file_name, label
            )
            os.makedirs(os.path.dirname(self.output_data.csv_path), exist_ok=True)
            print(f"Saving to {self.output_data.csv_path}")
            print(i)
            # reset df
            self.output_data.data_frame = None
            countdown_timer(2)
            start_time = datetime.now()
            finish_time = start_time + run_time
            current_time = datetime.now()
            measurement_number = 0
            while current_time < finish_time:
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                if print_countdown:
                    print(f"Running for another {(run_time - elapsed_time)}")
                self.take_measurement(
                    s_params_measure,
                    s_params_output,
                    elapsed_time,
                    label,
                    id=start_time.strftime(DateFormats.CURRENT.value),
                )
                measurement_number += 1
                if measurement_number % 10 == 0:
                    self.output_data.data_frame.to_csv(
                        self.output_data.csv_path, index=False
                    )

        self.output_data.data_frame.to_csv(self.output_data.csv_path, index=False)

        self.vna_object.CloseVNA()
        print("VNA Closed")
        return self.output_data