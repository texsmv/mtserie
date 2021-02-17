import numpy as np
from .mtserie import MTSerie
from numpy import unique
from .distances import DistanceType
from .projections import distance_matrix, mds_projection
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

from core import mtserie

class MTSerieDataset:
    """summary for [MTSerieDataset]
    
        This class assumes that the [MTSerie] objects use the same
        variables, although they can be uneven or misaligned with 
        respect to time
    """
    
    @property
    def distanceMatrix(self) -> np.ndarray:
        return self._distanceMatrix
    
    @distanceMatrix.setter
    def distanceMatrix(self, value):
        self._distanceMatrix = value
    
    @property
    def distanceMatrix_k(self) -> list:
        return self._distanceMatrix_k
    
    @distanceMatrix_k.setter
    def distanceMatrix(self, value):
        self._distanceMatrix_k = value
    
    @property
    def ids(self) -> list:
        return list(self.mtseries.keys())
    
    @property
    def first(self) -> MTSerie:
        return self.mtseries[next(iter(self.mtseries))]
    
    @property
    def temporalVariables(self):
        if self._isDataUniformInVariables:
            return self.first.labels
        return [mtserie.labels for mtserie in self.mtseries.values()]
    
    @property
    def variablesLen(self) -> int:
        if self._isDataUniformInVariables:
            return self.first.variablesLen
        return [mtserie.variablesLen for mtserie in self.mtseries.values()]
    
    @property
    def timeLen(self) -> int:
        if self._isDataUniformInTime:
            return self.first.timeLen
        return [mtserie.timeLen for mtserie in self.mtseries.values()]
    
    @property 
    def instanceLen(self):
        return len(self.mtseries)
    
    @property
    def categoricalLabels(self) -> list:
        return self.first.categoricalLabels
    
    @property
    def numericalLabels(self) -> list:
        return self.first.numericalLabels
    
    def __init__(self):
        self.mtseries = {}
        self.procesedMTSeries = {}
        self._isDataUniformInTime = True
        self._isDataUniformInVariables = True
        self._distanceMatrix = None
        self._distanceMatrix_k = None
        self._variablesLimits = {}
        self._projections = {}
        self._clusters = {}
        self._clusterById = {}
        
        super().__init__()

    def add(self, mtserie, identifier):
        assert isinstance(mtserie, MTSerie)
        assert isinstance(identifier, str)
        
        self.mtseries[identifier] = mtserie
        # * Added to procesed mtseries by reference
        self.procesedMTSeries[identifier] = mtserie
        
        if self._isDataUniformInVariables:
            self._isDataUniformInVariables = self.variablesLen == mtserie.variablesLen 
        
        if self._isDataUniformInTime:
            self._isDataUniformInTime = self.timeLen == mtserie.timeLen
        
        assert self.categoricalLabels == mtserie.categoricalLabels
        assert self.numericalLabels == mtserie.numericalLabels
    
    def get_mtseries(self, ids = [], procesed = True):
        if len(ids) == 0:
            if procesed:
                return list(self.procesedMTSeries.values())
            else:
                return list(self.mtseries.values())
        else:
            if procesed:
                return [self.procesedMTSeries[id] for id in ids]
            else:
                return [self.mtseries[id] for id in ids]
    
    def get_mtserie(self, id, procesed = True):
        if procesed:
            return self.procesedMTSeries[id]
        else:
            return self.mtseries[id]
    
    
    
    def compute_distance_matrix(self, variables = [], alphas = [], distanceType = DistanceType.EUCLIDEAN, L = 10, procesed = True):
        '''
        Implementation of distance matrix defined in "Interactive visualization of multivariate time series data"

        Args:
            distanceType (DistanceType, optional): Distance to compare mtseries. Defaults to DistanceType.EUCLIDEAN.
            L (int, optional): Window size used for MPdist. Defaults to 10.
        '''
        _variables = variables
        if len(variables) == 0: 
            _variables = self.temporalVariables
        
        _alphas = alphas
        if len(alphas) == 0:
            _alphas = np.ones(len(_variables))
        assert len(_alphas) == len(_variables)
    
        self._distanceMatrix, self._distanceMatrix_k = distance_matrix(
            self.get_mtseries(procesed=procesed), variables=_variables, 
            alphas=_alphas, distanceType=distanceType
            )
    
    def compute_projection(self):
        coords = mds_projection(self._distanceMatrix)
        for i in range(self.instanceLen):
            self._projections[self.ids[i]] = coords[i]
        
    def cluster_projections(self, n_clusters):
        coords = np.array(list(self._projections.values()))
        
        # ! spectral clustering not working
        # clustering = SpectralClustering(n_clusters=40,
        #                         assign_labels="discretize",
        #                         n_neighbors=2,
        #                         random_state=0).fit(coords)
        # return clustering.labels_

        # * dbscan
        # print(coords.shape)
        # clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
        # fit model and predict clusters
        # clusters = clustering.labels_
        
        k_means = KMeans(random_state=0, n_clusters=n_clusters)
        k_means.fit(coords)
        labels = k_means.predict(coords)
        
        self._clusters = {}
        clusterLabels = np.unique(labels)
        for clusterLabel in clusterLabels:
            clusterIds = []
            for i in range(self.instanceLen):
                if labels[i] == clusterLabel:
                    clusterIds = clusterIds + [self.ids[i]]
                    self._clusterById[self.ids[i]] = clusterLabel
            self._clusters[clusterLabel] = clusterIds

    
    
    def query_all_by_range(self, begin, end, toList = False):
        assert self._isDataUniformInTime
        assert isinstance(toList, bool)
        result = {}
        for id, mtserie in self.mtseries.items():
            assert isinstance(mtserie, MTSerie)
            result[id] = mtserie.range_query(begin, end)
        return result
    
    # ! deprecated
    def getAllMetadata(self):
        result = {}
        for id, mtserie in self._timeSeries.items():
            assert isinstance(mtserie, MTSerie)
            result[id] = {'metadata': mtserie.metadata, 'numFeatures' : mtserie.numericalFeatures.tolist(), 'numLabels' : mtserie.numericalLabels, 'catFeatures' : mtserie.categoricalFeatures.tolist(), 'catLabels' : mtserie.categoricalLabels}
        return result
    
    # ! deprecated
    def computeVariablesLimits(self):
        self._variablesLimits = {}
        for varName in self._variablesNames:
            currMin = None
            currMax = None
            for mtserie in self._timeSeries.values():
                assert isinstance(mtserie, MTSerie)
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
    # ! deprecated
    def getVariablesLimits(self):
        return self._variablesLimits
    # ! deprecated
    def getVariableLimits(self, varName):
        return self._variablesLimits[varName]
    # ! deprecated
    def setVariableLimits(self, varName, minValue, maxValue):
        self._variablesLimits[varName] = [minValue, maxValue]
    
    def removeVariable(self, varName):
        for mtserie in self._timeSeries.values():
            assert isinstance(mtserie, MTSerie)
            mtserie.removeTimeSerie(varName)
        