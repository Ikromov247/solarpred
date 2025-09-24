"""
Custom Exceptions
"""

class TrainSizeError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message

class TestSizeError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message


class ValidationError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message


class DataProcessingError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message

class ModelTrainingError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message