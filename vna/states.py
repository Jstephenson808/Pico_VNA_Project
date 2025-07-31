from datetime import timedelta
from vna.state_machine import State
from vna.vna_experiment import VNAExperiment
from vna.PicoVNA2 import PicoVNA2
from vna.VNA_data import VnaData
from vna.VNA_calibration import VnaCalibration
from vna.VNA_enums import MeasureSParam, TwoPortSParams
from vna.s_parameter_data import SParameterData

from datetime import timedelta
import pandas as pd
from vna.state_machine import State
from vna.vna_experiment import VNAExperiment
from vna.PicoVNA2 import PicoVNA2
from vna.VNA_data import VnaData
from vna.VNA_calibration import VnaCalibration
from vna.VNA_enums import MeasureSParam, TwoPortSParams, DataFrameCols
from vna.s_parameter_data import SParameterData
from vna.ml_pipeline import create_ml_pipeline

from vna.data_sources import DataSource


class FeatureExtractionState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Feature Extraction...")
        # In a real pipeline, you would load your training data here.
        # For this example, we'll use the captured data as both the
        # training data and the data to be predicted.

        # The pipeline needs a target series (y) for fitting.
        # We'll create a dummy target series.
        df = experiment_data.raw_data.data_frame
        y_train = pd.Series(
            df[DataFrameCols.LABEL.value].unique(),
            index=df[DataFrameCols.ID.value].unique(),
        )

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

        experiment_data.classification_result = predictions[0]  # Example

        self.state_machine.transition_to(ClassificationState())


class ClassificationState(State):
    def execute(self, experiment_data: VNAExperiment) -> None:
        print("Executing Classification...")
        print(f"Classification Result: {experiment_data.classification_result}")
        self.state_machine.transition_to(None)  # Stop the machine


class DataCaptureState(State):
    """
    A generic state for capturing data from a data source.
    """

    def __init__(self, data_source: DataSource):
        super().__init__()
        self.data_source = data_source

    def execute(self, experiment_data: VNAExperiment) -> None:
        """
        Reads from the data source and populates the experiment_data object.
        """
        print("Executing Data Capture...")
        experiment_data.raw_data = self.data_source.read(experiment_data)
        print("Data capture complete.")
        self.state_machine.transition_to(FeatureExtractionState())
