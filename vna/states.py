from datetime import timedelta
from vna.state_machine import State
from vna.vna_experiment import VNAExperiment
from vna.VNA import VNA
from vna.VNA_data import VnaData
from vna.VNA_calibration import VnaCalibration
from vna.VNA_enums import MeasureSParam, TwoPortSParams
from vna.s_parameter_data import SParameterData

# Placeholder for the next state
class FeatureExtractionState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Feature Extraction...")
        # In a real implementation, this would process experiment_data.raw_data
        # and populate experiment_data.processed_features
        self.state_machine.transition_to(ClassificationState())

# Placeholder for the classification state
class ClassificationState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Classification...")
        # This would use experiment_data.processed_features to classify the gesture
        # and populate experiment_data.classification_result
        experiment_data.classification_result = "Example Gesture"
        self.state_machine.transition_to(EndState())

# Final state to end the process
class EndState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Workflow finished.")
        print(f"Classification Result: {experiment_data.classification_result}")
        self.state_machine.transition_to(None) # Stop the machine

class DataCaptureState(State):
    """
    A state for capturing data from the VNA.
    """
    def __init__(self, calibration_path: str, run_time: timedelta, label: str):
        super().__init__()
        self.calibration_path = calibration_path
        self.run_time = run_time
        self.label = label

    def execute(self, experiment_data: VNAExperiment) -> None:
        """
        Connects to the VNA, captures data, and populates the experiment_data object.
        """
        print("Executing Data Capture...")
        
        # 1. Setup VNA and Data objects
        calibration = VnaCalibration(self.calibration_path)
        vna_data = VnaData(s_params_to_save=[TwoPortSParams.S21])
        vna = VNA(calibration=calibration, vna_data=vna_data)

        # 2. Connect and load calibration
        vna.connect()
        vna.load_cal()

        # 3. Perform the measurement (simplified loop from VNA.py)
        # In a real scenario, you would adapt the full loop from measure_n_times
        vna.take_measurement(
            s_params_measure=MeasureSParam.S21,
            s_params_output=[TwoPortSParams.S21],
            elapsed_time=timedelta(seconds=0), # Example value
            label=self.label,
            id=experiment_data.experiment_id
        )
        
        vna.close_connection()

        # 4. Convert the captured data into our common data format
        df = vna_data.dict_list_to_df()
        experiment_data.raw_data = SParameterData(df)

        print("Data capture complete.")

        # 5. Transition to the next state
        self.state_machine.transition_to(FeatureExtractionState())
