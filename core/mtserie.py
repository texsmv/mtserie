from numpy.lib.arraysetops import isin
from .utils import allowed_downsample_rule
import pandas as pd
import numpy as np
import copy
from .utils import is_array_like, to_np_array
from enum import Enum

class IndexType(Enum):
    INT = 0
    CATEGORICAL = 1
    DATETIME = 2


class MTSerie:
    """
    A class used to represent multivariate time series

    ...

    Attributes
    ----------
    variables : dict of np.ndarray<float>
        The variables in this dict are assumed to be time-dependent , thus, they are 
        represented as arrays of float values.
        a dict on numpy arrays of type float and length T. The number of elements
        in [variables] is D which is the number of variables. 
    variablesDates : dict of np.ndarray<DateTime>
        These dict is used in case [isDataDated] and [isDataDatedPerVariables] are true.
        a dict on numpy arrays of type DateTime and length T. The number of elements
        in [variables] is D which is the number of variables.
        It is assumed that [isDataEven] and [isDataAligned] is true. Otherwise, it will
        be empty
        
    variablesNames : list of str
        Names of the time dependent variables. These are used to make the queries in 
        [variables] and [variablesDates].
    
    dates : np.ndarray<DateTime>
        This array is used in case [isDataDated] is true and [isDataDatedPerVariables]
        is false.
        dates of the time dependent variables. Its length is equal to the length
        of each variable array in [variables].
        It is assumed that [isDataEven] and [isDataAligned] is true. Otherwise, it will
        be empty.
        
    isDataDated: boolean 
        True if time-dependent variables are dated either for each variable or for all
        variables. 
    
    isDataDatedPerVariable: boolean 
        True if time-dependent variables are dated per each variable. Otherwise all 
        time-dependent variables share same dates.
    
    isDataEven: boolean
        True if all time-dependent variables data have the same length
    
    isDataAligned: boolean
        True if all time-dependent variables data have the same length and share the 
        same dates. 
        In other words true if ([isDataEven and isDataDated and !isDataDatedPerVariable])
        
    isAnyVariableNamed: boolean
        True if a list of str of names is given to identify each time-dependent variable
        Otherwise the names in are given base on its index e.g: 0, 1, 2 ....
    
    timeLength: float or a list of floats
        if [isEven] is true then it returns the length of the time-dependent series,
        otherwise, it returns a list of floats with the length of each time serie
    
    variablesLength: float
        returns the number of time-dependent variables
    
    """
    
    
    @property
    def timeLen(self) -> int:
        return len(self.dataframe)

    @property
    def variablesLen(self) -> int:
        return len(self.dataframe.columns)

    @property
    def indexType(self):
        return self._indexType
    
    @indexType.setter
    def indexType(self, value):
        self._indexType = value
        
    @property
    def isDataDated(self) -> bool:
        return self._indexType == IndexType.DATETIME
    
    @property
    def labels(self) -> list:
        return self.dataframe.columns.tolist()
    
    @property
    def indexTypeStr(self) -> str:
        if self._indexType == IndexType.INT:
            return "intIndex"
        elif self._indexType == IndexType.CATEGORICAL:
            return "categoricalIndex"
        elif self._indexType == IndexType.DATETIME:
            return "datetimeIndex"
    
    @property
    def datetimes(self) -> np.ndarray:
        if not self.isDataDated:
            return None
        return self.dataframe.index.to_numpy()
        
    @property
    def datetimeLimits(self) -> list:
        if not self.isDataDated:
            return None
        return [self.dataframe.index[0].to_numpy(), self.dataframe.index[-1].to_numpy()]
    
    @property
    def categoricalLabels(self) -> list:
        return list(self.categoricalFeatures.keys())
    
    @property
    def numericalLabels(self) -> list:
        return list(self.numericalFeatures.keys())
    
    def __str__(self):
        return str(self.dataframe) + "\n\n" + "index type: " + self.indexTypeStr

    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.info = {}
        self.categoricalFeatures = {}
        self.numericalFeatures = {}
        self._indexType = IndexType.INT
        
        super().__init__()
    
    def range_query(self, begin, end):
        mask = (self.dataframe.index >= begin) & (self.dataframe.index < end)
        queryMTSerie = MTSerie()
        queryMTSerie.dataframe = self.dataframe[mask]
        queryMTSerie.info = self.info
        queryMTSerie.categoricalFeatures = self.categoricalFeatures
        queryMTSerie.numericalFeatures = self.numericalFeatures
        queryMTSerie.indexType = self.indexType
        return queryMTSerie

    def resample(self, rule):
        assert self.isDataDated
        downsampledMTSerie = self.clone()
        downsampledMTSerie.dataframe = self.dataframe.resample(rule).mean()
        return downsampledMTSerie
    
    def downsample_rules(self) -> list:
        return allowed_downsample_rule(self.dataframe)

    def clone(self):
        mtserie = MTSerie()
        assert isinstance(mtserie, MTSerie)
        mtserie.dataframe = self.dataframe.copy(deep=True)
        mtserie.info = copy.deepcopy(self.info)
        mtserie.numericalFeatures = copy.deepcopy(self.numericalFeatures)
        mtserie.categoricalFeatures = copy.deepcopy(self.categoricalFeatures)
        mtserie.indexType = self.indexType
        return mtserie
    
    def get_serie(self, label):
        return self.dataframe[label].to_numpy()
    
    def remove_serie(self, label):
        del self.dataframe.columns[label]
    
    def zNormalize(self, labels = []):
        _labels = labels
        if len(labels) == 0:
            _labels = self.labels
        for label in _labels:
            self.dataframe[label] = (self.dataframe[label] - self.dataframe[label].mean()) / self.dataframe[label].std(ddof=0)
    
        
    # !deprecated
    def normalize_data(self):
        for variableName in self.labels:
            x = self.tseries[variableName]
            self.tseries[variableName] = (x-min(x))/(max(x)-min(x))
    
    def plot(self, labels = None):
        if is_array_like(labels):
            self.dataframe[labels].plot()
        else:
            self.dataframe.plot()
    
    @staticmethod 
    def fromDArray(X, index = [], labels = [], info = {}, categoricalFeatures = {}, numericalFeatures = {}):
        assert is_array_like(X)
        assert is_array_like(index)
        assert is_array_like(labels)
        mtserie = MTSerie()
        
        assert isinstance(mtserie, MTSerie)
        
        _labels = []
        _data = None
        _index = []
            
        _data = to_np_array(X).transpose()
        
        if len(labels) != 0:
            assert (_data.shape[1] == len(labels))
            _labels = labels
        else:
            _labels = np.array([str(i) for i in range(len(X))])
        
        if len(index) != 0:
            _index = to_np_array(index)
            if type(_index[0]) == np.datetime64:
                mtserie.indexType = IndexType.DATETIME
            else:
                mtserie.indexType = IndexType.CATEGORICAL
        else:
            mtserie.indexType = IndexType.INT
            _index = np.array(range(_data.shape[0]))
        
        mtserie.dataframe = pd.DataFrame(data=_data, columns=_labels, 
                                         index=_index)
        mtserie.info = info
        mtserie.categoricalFeatures = categoricalFeatures
        mtserie.numericalFeatures = numericalFeatures
        
        return mtserie
    
    @staticmethod 
    def fromDict(X, index = [], info = {}, categoricalFeatures = {}, numericalFeatures = {}):
        assert isinstance(X, dict)
        assert is_array_like(index)
        mtserie = MTSerie()
        
        assert isinstance(mtserie, MTSerie)
        
        _labels = []
        _data = None
        _index = []
            
        _data = to_np_array(list(X.values())).transpose()
        
        _labels = np.array(list(X.keys()))
        
        if len(index) != 0:
            _index = to_np_array(index)
            if type(_index[0]) == np.datetime64:
                mtserie.indexType = IndexType.DATETIME
            else:
                mtserie.indexType = IndexType.CATEGORICAL
        else:
            mtserie.indexType = IndexType.INT
            _index = np.array(range(_data.shape[0]))
        
        mtserie.dataframe = pd.DataFrame(data=_data, columns=_labels, 
                                         index=_index)
        mtserie.info = info
        mtserie.categoricalFeatures = categoricalFeatures
        mtserie.numericalFeatures = numericalFeatures
        
        return mtserie
    