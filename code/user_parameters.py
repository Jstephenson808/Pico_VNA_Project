# GESTURE CAPTURE
from datetime import timedelta

ANTENNA_LABEL = "Test_dipole2"
#multiple test gestures; list all labels
TEST_GESTURES = ["A", "B", "C"]
#number of tests to be applied for each label in TEST_GESTURES
NUMBER_OF_TESTS = 10
TEST_TIME = timedelta(seconds=2)

# CALIBRATION
# This file is assumed to be in the "calibration" folder
CALIBRATION_FNAME = "500MHz_3GHz_MiniCirc_P1Short_P2Long_m3dBm_Lab103_Mar23_200MHz_6GHz_.cal"
