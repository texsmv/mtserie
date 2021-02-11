import numpy as np
import json


def is_array_like(a):
    """
    Helper function to determine if a value is array like.
    Parameters
    ----------
    a : obj
        Object to test.
    Returns
    -------
    True or false respectively.
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """
    return isinstance(a, (list, tuple, np.ndarray))


def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.
    Parameters
    ----------
    a : Tuple, list or np.ndarray
        The object to transform.
    Returns
    -------
    The np.ndarray.
    Raises
    ------
    ValueError
        If a is not a valid type.
        
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)


def clean_nan_inf(ts):
    """
    Converts tuples & lists to Numpy arrays and replaces nan and inf values with zeros

    Parameters
    ----------
    ts: Time series to clean
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    #Convert time series to a Numpy array
    ts = to_np_array(ts)

    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    return ts


def zNormalize(ts):
    """
    Returns a z-normalized version of a time series.

    Parameters
    ----------
    ts: Time series to be normalized
    
    from https://github.com/matrix-profile-foundation/matrixprofile repository
    """

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std

    return ts

# todo: document this
def mtserieQueryToJsonStr(query):
    assert isinstance(query, dict)
    if isinstance(next(iter(query.values())), np.ndarray):
        newQuery = {}
        for id, series in query.items():
            newQuery[id] = series.tolist()
        return json.dumps(newQuery)
    return json.dumps(query)

""" 
    upon request ranks the time series according to how well each time series separates those subsets.
    
    D_list: list of distance matrix D^2_k
"""
def subsetSeparationRanking(D_list, u_ind, v_ind):
    n = len(u_ind)
    m = len(v_ind)
    js = []
    for D_k in D_list:
        firstTerm = 0
        for i in u_ind:
            for j in v_ind:
                firstTerm = firstTerm + D_k[i][j]
        firstTerm =  firstTerm / (n * m)
        
        s_u = 0
        secondTerm = 0
        for i in u_ind:
            for j in u_ind:
                secondTerm = secondTerm + D_k[i][j]
        s_u = secondTerm / (2 * n)
        secondTerm =  secondTerm / (2 * n * n)
        
        s_v = 0
        thirdTerm = 0
        for i in v_ind:
            for j in v_ind:
                thirdTerm = thirdTerm + D_k[i][j]
        s_v = thirdTerm / (2 * m)
        thirdTerm =  thirdTerm / (2 * m * m)
        
        
        num = firstTerm - secondTerm - thirdTerm
        
        den = s_u + s_v
        
        print("num: " + str(num) + " den: " + str(den))
        
        
        j_k = num / (den + 1e-17)
        
        js = js + [j_k]
    return js


def allowed_downsample_rule(df):
    timedelta = df.index.to_series().diff().median().to_numpy()
    
    years = timedelta.astype('timedelta64[Y]') / np.timedelta64(1, 'Y')
    months = timedelta.astype('timedelta64[M]') / np.timedelta64(1, 'M')
    days = timedelta.astype('timedelta64[D]') / np.timedelta64(1, 'D')
    hours = timedelta.astype('timedelta64[h]') / np.timedelta64(1, 'h')
    minutes = timedelta.astype('timedelta64[m]') / np.timedelta64(1, 'm')
    seconds = timedelta.astype('timedelta64[s]') / np.timedelta64(1, 's')
    nanoSeconds = timedelta.astype('timedelta64[ns]') / np.timedelta64(1, 'ns')
    
    if years != 0:
        return []
    
    if months != 0:
        return ['Y']

    if days != 0:
        return ['Y', 'M']

    if hours != 0:
        return ['Y', 'M', 'D']
    
    if minutes != 0:
        return ['Y', 'M', 'D', 'h']
    
    if seconds != 0:
        return ['Y', 'M', 'D', 'h', 'm']

    if nanoSeconds != 0:
        return ['Y', 'M', 'D', 'h', 'm', 's']

    return []
    