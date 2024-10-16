from enum import Enum

class MeasurementKey(Enum):
    PHASE = "phase"
    MAGNITUDE = "magnitude"
    BOTH = "both"

class ConfusionMatrixKey(Enum):
    FILTERED_DT = 'filtered_dt_confusion_matrix'
    FULL_DT = 'full_dt_confusion_matrix'
    FULL_SVM = 'full_svm_confusion_matrix'
    FILTERED_SVM = 'filtered_svm_confusion_matrix'

class ConfusionMatrixKey(Enum):
    FILTERED_DT = 'filtered_dt_confusion_matrix'
    FULL_DT = 'full_dt_confusion_matrix'
    FULL_SVM = 'full_svm_confusion_matrix'
    FILTERED_SVM = 'filtered_svm_confusion_matrix'

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

class MeasureSParam(Enum):
    S11 = "S11"
    S21 = "S21"
    S11_S21 = "S11+S21"
    ALL = "All"


class SParam(Enum):
    S11 = "S11"
    S12 = "S12"
    S22 = "S22"
    S21 = "S21"


class MeasurementFormat(Enum):
    LOGMAG = "logmag"
    PHASE = "phase"
    REAL = "real"
    IMAG = "imag"
    SWR = "swr"
    GROUP_DELAY = "gd"
    TIME_DOMAIN = "td"


class DataFrameCols(Enum):
    TIME = "time"
    S_PARAMETER = "s_parameter"
    FREQUENCY = "frequency"
    MAGNITUDE = "magnitude"
    PHASE = "phase"
    LABEL = "label"
    ID = "id"
