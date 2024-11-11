from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any


class NDArray(ABC):
    """
    Abstract class for n-dimensional arrays
    """

    @property
    @abstractmethod
    def shape(self) -> int | tuple[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> list[int] | list[list[int]]:
        raise NotImplementedError

    @abstractmethod
    def flatten(self) -> "NDArray1":
        raise NotImplementedError

    @abstractmethod
    def reshape(self, n: int, m: int) -> "NDArray2":
        raise NotImplementedError

class NDArray1(NDArray):

    def __init__(self, data: list[int]):
        self._data = data
        self._shape = len(data)

    def __add__(self, other: Any) -> "NDArray1":
        if isinstance(other, int):
            return NDArray1([x + other for x in self._data])
        elif isinstance(other, NDArray1):
            if self.shape != other.shape:
                raise ValueError("Shapes do not match for addition")
            return NDArray1([x + y for x, y in zip(self._data, other.data)])
        else:
            raise ValueError("Addition with type not supported")

    def __gt__(self, other: Any) -> list[bool]:
        if isinstance(other, int):
            return [x > other for x in self._data]
        else:
            raise ValueError("Comparison with type not supported")

    def __contains__(self, other: Any) -> bool:
        if isinstance(other, int):
            return other in self._data
        else:
            raise ValueError("Contains check with type not supported")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NDArray1):
            return self._data == other.data
        else:
            return False

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def data(self) -> list[int]:
        return self._data

    def flatten(self) -> "NDArray1":
        return self

    def reshape(self, n: int, m: int) -> "NDArray2":
        if n * m != self._shape:
            raise ValueError("Cannot reshape array of size {} into shape ({}, {})".format(self._shape, n, m))
        reshaped_data = [self._data[i * m:(i + 1) * m] for i in range(n)]
        return NDArray2(reshaped_data)

class NDArray2(NDArray):

    def __init__(self, data: list[list[int]]):
        if any(len(row) != len(data[0]) for row in data):
            raise ValueError("All rows must have the same length")
        self._data = data
        self._shape = (len(data), len(data[0]))

    def __add__(self, other: Any) -> "NDArray2":
        if isinstance(other, int):
            return NDArray2([[x + other for x in row] for row in self._data])
        elif isinstance(other, NDArray2):
            if self.shape != other.shape:
                raise ValueError("Shapes do not match for addition")
            return NDArray2([[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(self._data, other.data)])
        else:
            raise ValueError("Addition with type not supported")

    def __gt__(self, other: Any) -> list[list[bool]]:
        if isinstance(other, int):
            return [[x > other for x in row] for row in self._data]
        else:
            raise ValueError("Comparison with type not supported")

    def __contains__(self, other: Any) -> bool:
        if isinstance(other, int):
            return any(other in row for row in self._data)
        else:
            raise ValueError("Contains check with type not supported")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NDArray2):
            return self._data == other.data
        else:
            return False

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def data(self) -> list[list[int]]:
        return self._data

    def flatten(self) -> "NDArray1":
        flattened_data = [item for sublist in self._data for item in sublist]
        return NDArray1(flattened_data)

    def reshape(self, n: int, m: int) -> "NDArray2":
        flat_data = self.flatten().data
        if n * m != len(flat_data):
            raise ValueError("Cannot reshape array of size {} into shape ({}, {})".format(len(flat_data), n, m))
        reshaped_data = [flat_data[i * m:(i + 1) * m] for i in range(n)]
        return NDArray2(reshaped_data)