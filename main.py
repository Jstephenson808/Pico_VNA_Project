import time
from datetime import timedelta
import pandas as pd
from vna.state_machine import StateMachine
from vna.states import DataCaptureState
from vna.vna_experiment import VNAExperiment
from vna.data_sources import VNADatasource, FileDataSource

if __name__ == "__main__":
    # 1. Create the main data object for the experiment
    experiment_data = VNAExperiment(
        experiment_id=f"exp_{int(time.time())}",
        timestamp=pd.Timestamp.now(),
        metadata={"user": "test_user"}
    )

    # 2. Choose your data source
    #
    # Option A: Live data from the VNA
    # calibration_file = "calibrations/R505_MiniCirc_3dBm_MiniCirc1m_10Mto6G.cal"
    # data_source = VNADatasource(
    #     calibration_path=calibration_file,
    #     run_time=timedelta(seconds=5),
    #     label="test_gesture"
    # )
    #
    # Option B: Load data from a file (for testing/development)
    # Make sure to replace with a real file path
    data_file = "path/to/your/data.pkl" 
    data_source = FileDataSource(file_path=data_file, label="test_from_file")


    # 3. Define the initial state
    initial_state = DataCaptureState(data_source)

    # 4. Initialize and run the state machine
    state_machine = StateMachine(initial_state, experiment_data)
    state_machine.run()

    # 5. (Optional) Inspect the final data
    print("\n--- Experiment Summary ---")
    print(f"ID: {experiment_data.experiment_id}")
    print(f"Timestamp: {experiment_data.timestamp}")
    if experiment_data.raw_data:
        print(f"Raw data points: {len(experiment_data.raw_data.data_frame)}")
    if experiment_data.classification_result:
        print(f"Result: {experiment_data.classification_result}")
