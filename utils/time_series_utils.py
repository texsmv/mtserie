import xml.etree.ElementTree as ET 
import sys
import datetime
import numpy as np
from tslearn.metrics import dtw
import json
from ..classes.time_serie import MultivariateTimeSerie

sys.path.append("../../")
sys.path.append("../")
sys.path.append("..")
#from classes.time_serie import MultivariateTimeSerie


def tserieEuclideanDistance(x_1, x_2):
    # return (np.power(np.power(x_1 - x_2, 2).sum(), 1/2))
    return (np.power(np.power(x_1 - x_2, 2).sum(), 1/2)) / float(len(x_1))

def tserieDtwDistance(x_1, x_2):
    return dtw(x_1, x_2)


def euclideanDistance(m_1, m_2):
    return pow((m_1 - m_2) ** 2, 1/2.0)


def mtserieQueryToJsonStr(query):
    assert isinstance(query, dict)
    if isinstance(next(iter(query.values())), np.ndarray):
        newQuery = {}
        for id, series in query.items():
            newQuery[id] = series.tolist()
        return json.dumps(newQuery)
    return json.dumps(query)



def distance_matrix(X, alphas, metadata = np.array([]), metadataAlphas = []):
    assert isinstance(X, np.ndarray)
    assert isinstance(metadata, np.ndarray)
    
    print(alphas)
    md = 0
    n, d, t = X.shape
    
    includeMetadata = False
    
    if len(metadata) != 0:
        includeMetadata = True
        md = metadata.shape[1]
        assert metadata.shape[0] == n
    
    D_k = np.zeros([d, n, n])
    
    for k in range(d):
        for i in range(n):
            for j in range(n):
                # D_k[k][i][j] = tserieDtwDistance(X[i][k], X[j][k])
                D_k[k][i][j] = tserieEuclideanDistance(X[i][k], X[j][k])
    
    for k in range(d):
        D_k[k] = np.power(D_k[k], 2) * (alphas[k] ** 2)
        
    D = np.sum(D_k, axis=0)
    
    
    
    
    MD_k = []
    MD = []
    if(includeMetadata):
        MD_k = np.zeros([md, n, n])
        for k in range(md):
            for i in range(n):
                for j in range(n):
                    MD_k[k][i][j] = euclideanDistance(metadata[i][k], metadata[j][k])
    
        for k in range(md):
            MD_k[k] = np.power(MD_k[k], 2) * (metadataAlphas[k] ** 2)
            
        MD = np.sum(MD_k, axis=0)
        
        D = D.__add__(MD)
    
    D = np.power(D, 1/2)
    
    return D
    