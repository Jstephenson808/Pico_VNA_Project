from typing import List, Union

from VNA_enums import FourPortSParams


class SParameterCombinationsList:
    """
    Just a simple class to generate S Parameter lists from both strings and Enums
    """

    def __init__(self, param_list:List[List[Union[FourPortSParams, str]]]):
        self.list = self.list = [
            param if isinstance(param, FourPortSParams)
            else FourPortSParams[param.upper()]
            for s_param_list in param_list
            for param in s_param_list
        ]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.list):
            result = self.list[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
