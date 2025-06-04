from enum import Enum, StrEnum, IntEnum


class MeasurementKey(Enum):
    PHASE = "phase"
    MAGNITUDE = "magnitude"
    BOTH = "both"


class ConfusionMatrixKey(Enum):
    FILTERED_DT = "filtered_dt_confusion_matrix"
    FULL_DT = "full_dt_confusion_matrix"
    FULL_SVM = "full_svm_confusion_matrix"
    FILTERED_SVM = "filtered_svm_confusion_matrix"


class ConfusionMatrixKey(Enum):
    FILTERED_DT = "filtered_dt_confusion_matrix"
    FULL_DT = "full_dt_confusion_matrix"
    FULL_SVM = "full_svm_confusion_matrix"
    FILTERED_SVM = "filtered_svm_confusion_matrix"


class Movements(Enum):
    BEND = "bend"


class DfFilterOptions(Enum):
    PHASE = "phase"
    MAGNITUDE = "magnitude"
    BOTH = "both"


class DateFormats(Enum):
    CURRENT = "%Y_%m_%d_%H_%M_%S"
    ORIGINAL = "%Y_%m_%d_%H_%M_%S.%f"
    DATE_FOLDER = "%Y_%m_%d"
    MILLISECONDS = "%Y_%m_%d_%H_%M_%S.%f"
    VNA_FOLDER_DATE_FROMAT = "%y%m%d%H%M"


class TwoPortSParams(Enum):
    S11 = "S11"
    S12 = "S12"
    S22 = "S22"
    S21 = "S21"


class FourPortSParams(Enum):
    S11 = "S11"
    S12 = "S12"
    S13 = "S13"
    S14 = "S14"
    S21 = "S21"
    S22 = "S22"
    S23 = "S23"
    S24 = "S24"
    S31 = "S31"
    S32 = "S32"
    S33 = "S33"
    S34 = "S34"
    S41 = "S41"
    S42 = "S42"
    S43 = "S43"
    S44 = "S44"


class MagnitudeOrPhase(Enum):
    Magnitude = "magnitude"
    Phase = "phase"


class MeasurementFormat(Enum):
    LOGMAG = "logmag"
    PHASE = "phase"
    REAL = "real"
    IMAG = "imag"
    SWR = "swr"
    GROUP_DELAY = "gd"
    TIME_DOMAIN = "td"


class ClassificationResultsAccuracy(StrEnum):
    WEIGHTED_AVERAGE = "weighted avg"
    MACRO_AVERAGE = "macro avg"
    ACCURACY = "accuracy"


class DataFrameCols(Enum):
    TIME = "time"
    S_PARAMETER = "s_parameter"
    FREQUENCY = "frequency"
    MAGNITUDE = "magnitude"
    PHASE = "phase"
    LABEL = "label"
    ID = "id"


class ClassificationResultsColumns(StrEnum):
    LABEL = "label"
    CLASSIFIER = "classifier"
    FULL_OR_FILTERED = "full or filtered"
    TYPE = "type"
    S_PARAM = "s_param"
    LOW_FREQUENCY = "low_frequency"
    HIGH_FREQUENCY = "high_frequency"
    GESTURE = "gesture"
    PRECISION = "precision"
    RECALL = "recall"


class SParam2Port(Enum):
    S11 = "S11"
    S12 = "S12"
    S22 = "S22"
    S21 = "S21"


class MeasureSParam(Enum):
    S11 = "S11"
    S21 = "S21"
    S11_S21 = "S11+S21"
    ALL = "All"


class SnP(Enum):
    S1P = "S1P"
    S2P = "S2P"
    S3P = "S3P"
    S4P = "S4P"


class DfAxis(IntEnum):
    ROW = 0
    COLUMN = 1


if __name__ == "__main__":
    pass
