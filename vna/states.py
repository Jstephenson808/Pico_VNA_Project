from datetime import timedelta
from vna.state_machine import State
from vna.vna_experiment import VNAExperiment
from vna.VNA import VNA
from vna.VNA_data import VnaData
from vna.VNA_calibration import VnaCalibration
from vna.VNA_enums import MeasureSParam, TwoPortSParams
from vna.s_parameter_data import SParameterData

from datetime import timedelta
import pandas as pd
from vna.state_machine import State
from vna.vna_experiment import VNAExperiment
from vna.VNA import VNA
from vna.VNA_data import VnaData
from vna.VNA_calibration import VnaCalibration
from vna.VNA_enums import MeasureSParam, TwoPortSParams, DataFrameCols
from vna.s_parameter_data import SParameterData
from vna.ml_pipeline import create_ml_pipeline

class FeatureExtractionState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Feature Extraction...")
        # In a real pipeline, you would load your training data here.
        # For this example, we'll use the captured data as both the
        # training data and the data to be predicted.
        
        # The pipeline needs a target series (y) for fitting.
        # We'll create a dummy target series.
        df = experiment_data.raw_data.data_frame
        y_train = pd.Series(df[DataFrameCols.LABEL.value].unique(), index=df[DataFrameCols.ID.value].unique())

        # The pipeline also needs the main data (X) for fitting.
        X_train = pd.DataFrame(index=df[DataFrameCols.ID.value].unique())

        # Create the pipeline
        pipeline = create_ml_pipeline(timeseries_container=df)

        # Fit the pipeline
        print("Fitting the ML pipeline...")
        pipeline.fit(X_train, y_train)

        # Now, let's predict on the same data (for demonstration)
        print("Making predictions...")
        predictions = pipeline.predict(X_train)

        experiment_data.classification_result = predictions[0] # Example

        self.state_machine.transition_to(ClassificationState())

class ClassificationState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Classification...")
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
