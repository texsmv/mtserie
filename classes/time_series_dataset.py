from .time_serie import MultivariateTimeSerie
import numpy as np

class TimeSeriesDataset:
    """summary for [TimeSeriesDataset]
    
        This class assumes that the [MultivariateTimeSerie] objects use the same
        variables, although they can be uneven or misaligned with 
        respect to time
    """
    def __init__(self):
        self._timeSeries = {}
        
        self._timeLength = -1
        self._instanceLength = 0
        self._variablesLength = -1
        
        self._ids = []
        self._variablesNames = []
        self._categoricalLabels = []
        self._numericalLabels = []
        
        self.isDataEven = False
        self.isDataAligned = False
        
        self._variablesLimits = {}
        
        super().__init__()

    def getTimeLength(self):
        return self._timeLength

    def getInstanceLength(self):
        return self._instanceLength
    
    def getVariablesLength(self):
        return self._variablesLength
    
    def add(self, timeSerie, identifier):
        assert isinstance(timeSerie, MultivariateTimeSerie)
        assert isinstance(identifier, str)
        
        self._timeSeries[identifier] = timeSerie
        self._ids.append(identifier)
        self._instanceLength = self._instanceLength + 1
        
        if len(self._variablesNames)  == 0:
            self._variablesNames = timeSerie.getVariablesNames()
            self._variablesLength = timeSerie.variablesLength
        else:
            assert self._variablesNames == timeSerie.getVariablesNames()
        
        if len(self._categoricalLabels)  == 0:
            self._categoricalLabels = timeSerie.getCategoricalLabels()
        else:
            assert self._categoricalLabels == timeSerie.getCategoricalLabels()
        
        if len(self._numericalLabels) == 0:
            self._numericalLabels = timeSerie.getNumericalLabels()
        else:
            assert self._numericalLabels == timeSerie.getNumericalLabels()
            
        self.isDataEven = self.computeUniformity()
        self.isDataAligned = self.computeAlignment()
        
        # probably this should be computed later
        self._timeLength = self.computeTimeLength()
    
    def ids(self):
        return self._ids
    
    def getVariablesNames(self):
        return self._variablesNames
    
    def getTimeSerie(self, id):
        return self._timeSeries[id]
    
    def getValues(self):
        assert(self.isDataEven)
        assert(self.isDataAligned)
        X = []
        for id in self._ids:
            mtserie = self._timeSeries[id]
            assert isinstance(mtserie, MultivariateTimeSerie)
            x_i = []
            for dim in self._variablesNames:
                x_i.append(mtserie.getSerie(dim))
            x_i = np.array(x_i)
            X.append(x_i)
        return np.array(X)
    
    def getNumericalLabels(self):
        return self._numericalLabels
    
    def getCategoricalLabels(self):
        return self._categoricalLabels

    def getNumericalValues(self):
        # todo check this
        # assert(self.areNumericalFeaturesEven)
        M = []
        for id in self._ids:    
            mtserie = self._timeSeries[id]
            assert isinstance(mtserie, MultivariateTimeSerie)
            M.append(mtserie.getNumericalFeatures())
        return np.array(M)
    
    def getCategoricalValues(self):
        # todo check this
        # assert(self.areCategoricalFeaturesEven)
        M = []
        for id in self._ids:
            mtserie = self._timeSeries[id]
            assert isinstance(mtserie, MultivariateTimeSerie)
            M.append(mtserie.getCategoricalFeatures())
        return np.array(M)
    
    def computeUniformity(self):
        timeSeriesLength = next(iter(self._timeSeries.values())).timeLength
        if isinstance(timeSeriesLength, list):
            return False
        
        for (_, tserie) in self._timeSeries.items():
            assert isinstance(tserie, MultivariateTimeSerie)
            if not tserie.isDataEven:
                return False
            if isinstance(tserie.timeLength, list):
                return False
            if timeSeriesLength != tserie.timeLength:
                return False
        return True
    
    def computeAlignment(self):
        for (_, tserie) in self._timeSeries.items():
            assert isinstance(tserie, MultivariateTimeSerie)
            if not tserie.isDataAligned:
                return False
        return True
    
    def queryAllByIndex(self, beginIndex, endIndex, toList = False):
        assert self.isDataEven
        assert self.isDataAligned
        assert isinstance(toList, bool)
        result = {}
        for id, mtserie in self._timeSeries.items():
            assert isinstance(mtserie, MultivariateTimeSerie)
            result[id] = mtserie.queryByIndex(beginIndex, endIndex, toList=toList)
        return result

    def getAllMetadata(self):
        result = {}
        for id, mtserie in self._timeSeries.items():
            assert isinstance(mtserie, MultivariateTimeSerie)
            result[id] = {'metadata': mtserie.metadata, 'numFeatures' : mtserie.numericalFeatures.tolist(), 'numLabels' : mtserie.numericalLabels, 'catFeatures' : mtserie.categoricalFeatures.tolist(), 'catLabels' : mtserie.categoricalLabels}
        return result
    
    def computeTimeLength(self):
        
        if self.isDataEven and self.isDataAligned:
            return next(iter(self._timeSeries.values())).timeLength
        elif self.isDataAligned:
            return [e.timeLength for e in self._timeSeries.values()]
        
    def computeVariablesLimits(self):
        self._variablesLimits = {}
        for varName in self._variablesNames:
            currMin = None
            currMax = None
            for mtserie in self._timeSeries.values():
                assert isinstance(mtserie, MultivariateTimeSerie)
                minValue = mtserie.getSerie(varName).min()
                maxValue = mtserie.getSerie(varName).max()
                
                if(currMin == None):
                    currMin = minValue
                elif currMin > minValue:
                    currMin = minValue
                    
                if(currMax == None):
                    currMax = maxValue
                elif currMax < maxValue:
                    currMax = maxValue
            self._variablesLimits[varName] = [currMin, currMax]
    
    def getVariablesLimits(self):
        return self._variablesLimits

    def getVariableLimits(self, varName):
        return self._variablesLimits[varName]

    def setVariableLimits(self, varName, minValue, maxValue):
        self._variablesLimits[varName] = [minValue, maxValue]
        
    def removeVariable(self, varName):
        for mtserie in self._timeSeries.values():
            assert isinstance(mtserie, MultivariateTimeSerie)
            mtserie.removeTimeSerie(varName)
        
    # todo check utility
    # def areNumericalFeaturesEven(self):
    #     numericalFeaturesLength = len(next(iter(self._timeSeries.values())).numericalFeatures)
    #     for (_, tserie) in self._timeSeries.items():
    #         if numericalFeaturesLength != len(tserie.numericalFeatures):
    #             return False
            
    #     return True

    # def areCategoricalFeaturesEven(self):
    #     categoricalFeaturesLength = len(next(iter(self._timeSeries.values())).categoricalFeatures)
    #     for (_, tserie) in self._timeSeries.items():
    #         if categoricalFeaturesLength != len(tserie.categoricalFeatures):
    #             return False
        
    #     return True
            
    
    
