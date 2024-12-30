import os.path
from time import sleep

import pyvisa
from datetime import datetime, timedelta

from numba.cuda import runtime
from pyvisa.resources import MessageBasedResource
from scipiCommands import (
    set_snp_save_ports_command_string,
    create_directory_command_string,
    SnP,
    await_completion,
    load_state_command,
    save_snp_command_string,
)
from VNA_utils import countdown_timer


class ScipiGestureCaptureExperiment:
    def __init__(
        self,
        vna_handle: MessageBasedResource,
        test_gestures: [str],
        runtime: timedelta,
        experiment_countdown: timedelta,
        test_name: str,
        snp_format: SnP,
        root_folder: str = "local/James/Live_Captures",
        n_tests_per_gesture: int = 1,
        path_to_state_to_load: str = None,
    ):
        self.vna_handle: MessageBasedResource = vna_handle
        self.test_gestures: [str] = test_gestures
        self.run_time: timedelta = runtime
        self.experiment_countdown: timedelta = experiment_countdown
        self.test_name: str = test_name
        self.snp_format: SnP = snp_format
        self.root_folder: str = root_folder
        self.save_folder: str = None
        self.path_to_state_to_load: str = path_to_state_to_load
        self.n_tests_per_gesture: int = n_tests_per_gesture

    def run_gesture_capture(self):
        # load state if provided
        if self.path_to_state_to_load:
            self.vna_handle.write(load_state_command(self.path_to_state_to_load))

        self.set_touchstone_format()
        self.create_directories_for_exeriment()
        self.capture_gestures()

    def await_completion(self):
        while True:
            if int(self.vna_handle.query("*OPC?")) == 1:
                return

    def create_directories_for_exeriment(self):
        self.vna_handle.write(create_directory_command_string(self.root_folder))
        self.save_folder = f"{self.root_folder}/{datetime.now().strftime('%y%m%d%H%M')}_{self.test_name}"

        print(os.path.dirname(self.save_folder))
        self.vna_handle.write(create_directory_command_string(self.save_folder))

    def set_touchstone_format(self):
        self.vna_handle.write(set_snp_save_ports_command_string(self.snp_format))

    def capture_single_gesture(self, current_test_folder: str, test_number):
        self.vna_handle.write(
            save_snp_command_string(
                f"{current_test_folder}/capture_{str(test_number).zfill(5)}",
                self.snp_format,
            )
        )
        self.await_completion()

    def print_elapsed_time(self, run_time, current_time, start_time, test_index):
        elapsed_time = current_time - start_time
        print(
            f"Running for another {(run_time - elapsed_time)} test index={test_index}"
        )

    def capture_gestures(self):
        for gesture in self.test_gestures:

            print(f"Gesture: {gesture}")
            input("Press enter to continue...")

            current_gesture_folder = f"{self.save_folder}/Gesture_{gesture}"
            self.vna_handle.write(
                create_directory_command_string(current_gesture_folder)
            )
            for test_number in range(1, self.n_tests_per_gesture + 1):
                # create unique folder for test
                current_test_folder = f"{current_gesture_folder}/{test_number}"
                self.vna_handle.write(
                    create_directory_command_string(current_test_folder)
                )

                print(f"Test {test_number}")
                countdown_timer(self.experiment_countdown.total_seconds())
                start_time = datetime.now()
                finish_time = start_time + self.run_time

                current_time = datetime.now()
                test_index = 0
                while current_time < finish_time:
                    # # Trigger the instrument to start a sweep cycle
                    # SNA.write(':TRIGger:SEQuence:SINGle')
                    # Execute the *OPC? command and wait until the command returns 1 (the measurement cycle is completed).
                    self.capture_single_gesture(current_test_folder, test_index)

                    if test_index % 100 == 0:
                        self.print_elapsed_time(
                            self.run_time, current_time, start_time, test_index
                        )

                    current_time = datetime.now()
                    test_index += 1


def open_vna_handle(ni_visa_dll_path, vna_visa_address):
    rm = pyvisa.ResourceManager(ni_visa_dll_path)

    sna = rm.open_resource(vna_visa_address)
    print(f'Connected to {sna.query("*IDN?")}')
    return sna


if __name__ == "__main__":

    NI_VISA_DLL_PATH = r"C:\Windows\System32\nivisa64.dll"
    VNA_VISA_ADDRESS = "USB0::0xF4EC::0x1700::SNA5XCED5R0097::INSTR"

    COUNTDOWN_TIME = timedelta(seconds=3)
    TEST_NAME = f"liquid_metal_glove_6ges_25reps"
    snp = SnP.S4P
    STATE_PATH = "local/James/Calibration/glove_experiment_setup_201pts_100M_500M.csa"

    test_gestures = ["A", "B", "C", "1", "2", "3"]

    SAVE_ROOT = f"local/James/Live_Captures/{TEST_NAME}"

    vna_handle = open_vna_handle(NI_VISA_DLL_PATH, VNA_VISA_ADDRESS)

    RUN_TIME_DELTA = timedelta(seconds=5)
    n_tests = 25
    TEST_NAME = f"liquid_metal_glove_6ges_25rps"

    experiment = ScipiGestureCaptureExperiment(
        vna_handle,
        test_gestures,
        RUN_TIME_DELTA,
        COUNTDOWN_TIME,
        TEST_NAME,
        snp,
        SAVE_ROOT,
        n_tests,
        path_to_state_to_load=STATE_PATH,
    )
    experiment.run_gesture_capture()
