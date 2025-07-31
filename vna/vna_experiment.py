from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd
from vna.s_parameter_data import SParameterData

@dataclass
class VNAExperiment:
    """
    A container for all data related to a single VNA experiment,
    designed to be passed through a state machine.
    """
    # Metadata
    experiment_id: str
    timestamp: pd.Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    # State-specific data
    raw_data: Optional[SParameterData] = None
    processed_features: Optional[pd.DataFrame] = None
    classification_result: Optional[str] = None
    
    # You could add other fields as needed, e.g.:
    # error_message: Optional[str] = None
