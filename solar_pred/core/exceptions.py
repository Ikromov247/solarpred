"""
Custom Exceptions
"""

class TrainSizeError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message

class TestSizeErorr(Exception):
    def __init__(self, message: object) -> None:
        super().__init__(message)
        self.message = message