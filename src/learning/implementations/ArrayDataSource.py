import numpy as np
from typing import Generic, Type
from ...learning.definitions.DataSource import DataSourceType_co
from ...learning.definitions.SplittableDataSource import SplittableDataSource
from ...learning.definitions.ResettableDataSource import ResettableDataSource

DataSourceArray = np.ndarray[DataSourceType_co]
StaticDataSourceArray = np.ndarray[Type[DataSourceType_co]]

class ArrayDataSource(Generic[DataSourceType_co], SplittableDataSource[DataSourceType_co], ResettableDataSource[DataSourceType_co]):
    def __init__(self, data: DataSourceArray):
        self.__data = data
        self.__idx = 0
    
    def __len__(self) -> int:
        return len(self.__data)
    
    def shape(self) -> tuple[int, ...]:
        return self.__data.shape
    
    def get_next_batch(self, size: int = 1) -> DataSourceArray|None:
        # if we are not yet at the end, return the next `size` elements or whatever is left
        if self.__idx < len(self.__data):
            no_to_get = min(size, len(self.__data) - self.__idx)
            batch = self.__data[self.__idx:self.__idx + no_to_get]
            self.__idx += size
            return batch
        # otherwise, return None
        return None
    
    def get_all_data(self) -> DataSourceArray:
        return self.__data
    
    def reset(self) -> None:
        self.__idx = 0

    def split(self, splits: tuple[int, ...]) -> tuple[SplittableDataSource[DataSourceType_co], ...]:
        return self.split_from_array(self.__data, splits)
        
    
    @staticmethod
    def split_from_array(
        source: StaticDataSourceArray,
        splits: tuple[int, ...]
    ) -> tuple['ArrayDataSource', ...]:
        # ensure splits sums to one
        if sum(splits) != 1:
            raise ValueError('Splits must sum to 1')
        # ensure splits are in range
        if min(splits) <= 0 or max(splits) > 1:
            raise ValueError('Splits must be in range (0, 1]')
        # split data
        data_size = len(source)
        sizes_for_all_but_one = [int(data_size * split) for split in splits[:-1]]
        sizes = sizes_for_all_but_one + [data_size - sum(sizes_for_all_but_one)]
        data_source_instances = ()
        start = 0
        for size in sizes:
            data_source_instances += (ArrayDataSource(source[start:start + size]),)
            start += size
        return data_source_instances
    
    @staticmethod
    def from_arrays(
        sources: tuple[StaticDataSourceArray, StaticDataSourceArray]|tuple[StaticDataSourceArray, StaticDataSourceArray, StaticDataSourceArray]
    ) -> tuple['ArrayDataSource', ...]:
        # map to ArrayDataSource
         return tuple(ArrayDataSource(source) for source in sources)
    
        