class NotValidCalibrationFileException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotValidCSVException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotValidSParamException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FileNotInCorrectFolder(Exception):
    def __init__(self, message):
        super().__init__(message)


class VNAError(Exception):
    def __init__(self, message):
        super().__init__(message)
