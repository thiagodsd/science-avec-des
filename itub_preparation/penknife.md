# Functions

variable collector


```python
def var_cleaner(s):
    """
    ('var1, var2, ..., varN') -> None
    """
    trash = list()
    miss  = list()
    for v in s.replace(' ', '').split(','):
        if v in globals():
            del globals()[v]
            trash.append(v)
        else:
            miss.append(v)
    print('- DELETED:     {}'.format( ', '.join(trash) ))
    print('- NOT DEFINED: {}'.format( ', '.join(miss) ))
```

longitude and latitude distances to meters


```python
import numpy as np
from math import sin, cos, sqrt, atan2, radians
#
def lat_lon_converter(lat1, lon1, lat2, lon2, unit):
    """
    ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    """
    try:
        R = 6373.0
        dlon = radians(lon2) - radians(lon1)
        dlat = radians(lat2) - radians(lat1)
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        if unit == 'm':
            return distance * 10e3
        elif unit == 'km':
            return distance
    except ValueError:
        return np.nan
```

# Exploration

# Data Preparation

# Modelling

## Splitting


```python
import numpy as np

def split_train_test(data, test_ratio):
    """
    usage: train_set, test_set = split_train_test(housing, 0.2)
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size    = int( len(data) * test_ratio )
    test_indices     = shuffled_indices[ :test_set_size ]
    train_indices    = shuffled_indices[ test_set_size: ]
    
    return data.iloc[train_indices], data.iloc[test_indices]

```

## Techniques

### K-MEANS

### DBSCAN

dbscan approximate predicting


```python
def dbscan_predict(model, X):
    """
    ref: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    """
    nr_samples = X.shape[0]
    y_new      = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]   # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new
```

## Analysing


```python

```
