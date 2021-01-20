import numpy as np
import sys
import json

sys.path.append("..")

from .time_series_dataset import TimeSeriesDataset
from utils.time_series_utils import time_serie_from_eml_string, distance_matrix
from utils.time_series_utils import mtserieQueryToJsonStr
from dimensionality_reduction.mds import mts_mds

class EmotionDatasetController:
    def __init__(self):
        self.dataset =  TimeSeriesDataset()
        self.isCategorical = None
        self.minValue = None
        self.maxValue = None
        self.alphas = None
        self.numericalAlphas = None
        self.categoricalAlphas = None
        self.oldCoords = None
        super().__init__()
        
    def addEml(self, eml, isCategorical = True):
        if self.isCategorical is None:
            self.isCategorical = isCategorical
        else:
            assert self.isCategorical == isCategorical
            
        mtserie = time_serie_from_eml_string(eml, isCategorical= isCategorical)
        
        if self.alphas is None:
            self.alphas = np.ones(self.getVariablesNames())
        
        if self.categoricalAlphas is None:
            self.categoricalAlphas = np.ones(self.getVariablesNames())
        
        id = mtserie.metadata["id"]
        self.dataset.add(mtserie, id)
        return id
        
    def getIds(self):
        return self.dataset.ids()
    
    def calculateValuesBounds(self):
        X = self.dataset.getValues()
        assert isinstance(X, np.ndarray)
        self.minValue = X.min()
        self.maxValue = X.max()
        
    def getValuesBounds(self):
        if self.minValue != None and self.maxValue != None:
            return [self.minValue, self.maxValue]
        return [-1 ,-1]
    
    def setValuesBounds(self, minVal, maxVal):
        self.minValue = minVal
        self.maxValue = maxVal
    
    def getAllValuesInRange(self, begin, end):
        return self.dataset.queryAllByIndex(beginIndex=begin, endIndex=end, toList=True)
    
    def getTimeLength(self):
        return self.dataset.getTimeLength()
    
    def getInstanceLength(self):
        return self.dataset.getInstanceLength()
    
    def getVariablesNames(self):
        return self.dataset.getVariablesNames()
    
    def getNumericalLabels(self):
        return self.dataset.getNumericalLabels()

    def getCategoricalLabels(self):
        return self.dataset.getCategoricalLabels()
    
    def queryAllInRange(self, begin, end):
        return mtserieQueryToJsonStr(self.dataset.queryAllByIndex(begin, end, toList=True))
    
    def mdsProjection(self):
        X = self.dataset.getValues()
        Mnum = self.dataset.getNumericalValues()
        Mcat = self.dataset.getCategoricalValues()

        alphas = self.alphas
        numAlphas = self.numericalAlphas

        D = distance_matrix(X, alphas, metadata=Mnum, metadataAlphas= numAlphas)
        
        coords = mts_mds(D)
        
            
        if isinstance(self.oldCoords, np.ndarray): 
            P = coords
            Q = self.oldCoords
            A = P.transpose().dot(Q)
            u, s, vt = np.linalg.svd(A, full_matrices=True)
            v = vt.transpose()
            ut = u.transpose()
            r = np.sign(np.linalg.det(v.dot(ut)))
            R = v.dot(np.array([[1, 0], [0, r]])).dot(ut)

            print("sign: " + str(r))

            # print(u)
            # print(s)
            # print(vt)
            # print(r)
            # print(R)

            coords = R.dot(P.transpose()).transpose()
        
        coordsDict = {}
        ids = self.getIds()
        for i in range(len(ids)):
            id = ids[i]
            coord = coords[i]
            coordsDict[id] = coord.tolist()
        
        self.oldCoords = coords
        
        return json.dumps(coordsDict)