import os.path
from time import sleep

import pyvisa
from datetime import datetime, timedelta

from scipiCommands import save_snp_command_string
from scipiCommands import (
    set_snp_save_ports_command_string,
    create_directory_command_string,
    SnP,
)


def countdown_timer(seconds):
    while seconds > 0:
        print(f"{seconds}..")
        sleep(1)
        seconds -= 1
    print("Start")


NI_VISA_DLL_PATH = r"C:\Windows\System32\nivisa64.dll"
VNA_VISA_ADDRESS = "USB0::0xF4EC::0x1700::SNA5XCED5R0097::INSTR"


RUN_TIME_DELTA = timedelta(seconds=5)
COUNTDOWN_TIME_SECONDS = 1
TEST_NAME = f"auto_test"
SAVE_ROOT = (
    f"local/James/Live_Captures/{datetime.now().strftime('%y%m%d%H%M')}_{TEST_NAME}"
)
snp = SnP.S4P
n_tests = 2
test_gestures = ["1"]
# test_gestures = ["A", "B", "C", "1", "2", "3"]

rm = pyvisa.ResourceManager(NI_VISA_DLL_PATH)

SNA = rm.open_resource(VNA_VISA_ADDRESS)
print(f'Connected to {SNA.query("*IDN?")}')

SNA.write(set_snp_save_ports_command_string(snp))


# create dir -> unsure what happens if dir already exists? -> may be quite unsafe
print(os.path.dirname(SAVE_ROOT))
SNA.write(create_directory_command_string(os.path.dirname(SAVE_ROOT)))
SNA.write(create_directory_command_string(SAVE_ROOT))

# how can I get n points, freq range from the VNA?

for gesture in test_gestures:
    print(f"Gesture: {gesture}")
    input("Press enter to continue...")
    current_gesture_folder = f"{SAVE_ROOT}/Gesture_{gesture}"
    SNA.write(create_directory_command_string(current_gesture_folder))
    for test_number in range(1, n_tests + 1):
        # create unique folder for test
        current_test_foler = f"{current_gesture_folder}/{test_number}"
        SNA.write(create_directory_command_string(current_test_foler))

        print(f"Test {test_number}")
        countdown_timer(COUNTDOWN_TIME_SECONDS)
        run_time = RUN_TIME_DELTA
        start_time = datetime.now()
        start_time_string = start_time.strftime("%Y_%m_%d_%H_%M_%S")
        finish_time = start_time + run_time

        current_time = datetime.now()
        i = 0
        while current_time < finish_time:
            # # Trigger the instrument to start a sweep cycle
            # SNA.write(':TRIGger:SEQuence:SINGle')
            # Execute the *OPC? command and wait until the command returns 1 (the measurement cycle is completed).
            while True:
                if int(SNA.query("*OPC?")) == 1:
                    break
            if i % 100 == 0:
                elapsed_time = current_time - start_time
                print(f"Running for another {(run_time - elapsed_time)} i={i}")
            SNA.write(
                save_snp_command_string(
                    f"{current_test_foler}/capture_{str(i).zfill(5)}", SnP.S4P
                )
            )
            current_time = datetime.now()
            i += 1

# Preset the SNA again
# SNA.write(':SYSTem:PRESet')
#
# # Set data format to ASCII
# SNA.write(':FORMat:DATA ASC')
