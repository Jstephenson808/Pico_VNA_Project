import time
from datetime import datetime, timedelta
import pandas as pd
from vna.state_machine import StateMachine
from vna.states import DataCaptureState
from vna.vna_experiment import VNAExperiment

if __name__ == "__main__":
    # 1. Create the main data object for the experiment
    experiment_data = VNAExperiment(
        experiment_id=f"exp_{int(time.time())}",
        timestamp=pd.Timestamp.now(),
        metadata={"user": "test_user"}
    )

    # 2. Define the initial state
    # NOTE: Replace with a valid path to your calibration file
    calibration_file = "calibrations/R505_MiniCirc_3dBm_MiniCirc1m_10Mto6G.cal"
    initial_state = DataCaptureState(
        calibration_path=calibration_file,
        run_time=timedelta(seconds=5), # Example run time
        label="test_gesture"
    )

    # 3. Initialize and run the state machine
    state_machine = StateMachine(initial_state, experiment_data)
    state_machine.run()

    # 4. (Optional) Inspect the final data
    print("\n--- Experiment Summary ---")
    print(f"ID: {experiment_data.experiment_id}")
    print(f"Timestamp: {experiment_data.timestamp}")
    if experiment_data.raw_data:
        print(f"Raw data points: {len(experiment_data.raw_data.data_frame)}")
    if experiment_data.classification_result:
        print(f"Result: {experiment_data.classification_result}")
