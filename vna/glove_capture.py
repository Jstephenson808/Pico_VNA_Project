from datetime import timedelta

from vna.scipiCommands import SnP
from vna.scipiInteraction import open_vna_handle, ScipiGestureCaptureExperiment

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
