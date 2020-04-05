## bash

```sh
bundle exec jekyll serve
```

```sh
git pull; git add .; git commit -m 'minor changes'; git push
```

```sh
youtube-dl -citx --audio-format mp3 http://www.dailymotion.com/video/x578s92
```

## python

### kaggle dataset -> google colab

```sh
pip install -q kaggle
```

```py
from google.colab import drive, files
json = files.upload()
```

```sh
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

#kaggle competitions download -c nyc-taxi-trip-duration
kaggle datasets download thiagodsd/sao-paulo-metro
unzip '*.zip'
```

### ridge regression

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model    import Ridge
from sklearn.metrics         import mean_absolute_error

ridge_cv   = GridSearchCV(Ridge(fit_intercept=True), {'alpha': [2.37**i for i in range(-8, 8)]}, scoring='neg_mean_absolute_error', cv=5)
ridge_cv.fit(X, y)
print(ridge_cv.best_params_['alpha'])

ridreg = Ridge( alpha=ridge_cv.best_params_['alpha'], fit_intercept=True )
ridreg.fit(X, y)

display( mean_absolute_error( y, ridreg.predict(X) ) )
display( pd.Series(ridreg.coef_.flatten(), X.columns.values.flatten()).sort_values().plot(kind='bar') )
```


### logistic regression

polynomial

```python
%matplotlib inline
%config     InlineBackend.figure_format = 'retina'

from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

poly   = PolynomialFeatures(degree=7)
X_poly = poly.fit_transform(X)

C     = 1e-2
logit = LogisticRegression(C=C, random_state=17)
logit.fit(X_poly, y)
```

grid search cross-validation

```python
skf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
c_values       = np.logspace(-2, 3, 500)
logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=1, n_jobs=-1)

logit_searcher.fit(X_poly, y)

logit_searcher.C_
plt.plot(c_values, np.mean(logit_searcher.scores_[1], axis=0))
```


### decision trees

classifier

```python
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
clf_tree.fit(train_data, train_labels)
pred = clf_tree.predict(data)
```

regressor

```python
from sklearn.tree import DecisionTreeRegressor

reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)
reg_tree.fit(X_train, y_train)
reg_tree_pred = reg_tree.predict(X_test)
```


### svm

```python
from sklearn.svm             import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import f1_score

scores = list()

X = df.drop(labels=['target'], inplace=False, axis=1)
y = df['target']

for train_id, test_id in KFold(n_splits=10).split(df):
  X_train, X_test = X.iloc[train_id], X.iloc[test_id]
  y_train, y_test = y.iloc[train_id], y.iloc[test_id]

  svc = SVC(kernel='linear', C=1)
  svc.fit(X_train, y_train)
  
  # NAS PARTICOES DE TREINO !
  # y_pred = svc.predict(X_test)
  # scores.append(f1_score(y_test, y_pred))

  y_pred = svc.predict(X_train)
  scores.append(f1_score(y_train, y_pred))

display(scores)
display(np.mean(scores))
```

### knn


sempre importante normalizar tudo

```python
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler

knn    = KNeighborsClassifier(n_neighbors=10)
scaler = StandardScaler()

X_train_scaled   = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_holdout_scaled)
```

alternativamente

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler

knn_pipe   = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid   = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
knn_grid.fit(X_train, y_train)

print(knn_grid.best_params_, knn_grid.best_score_)
```


### naive bayes

```python
from sklearn.naive_bayes import GaussianNB    # continuous features

model = GaussianNB()
model.fit(X, y)
ynew = model.predict(Xnew)
```

exemplo de text classification e matriz de confusao

```python
from sklearn.naive_bayes             import MultinomialNB # discrete features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline                import make_pipeline
from sklearn.metrics                 import confusion_matrix

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

#

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square = True, annot  = True, fmt ='d', cbar=False,
            xticklabels = train.target_names, 
            yticklabels = train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
```

- - - 


- [Functions](#functions)
- [Exploration](#exploration)
- [Data Preparation](#data-preparation)
- [Modelling](#modelling)
	- [Splitting](#splitting)
	- [Techniques](#techniques)
		- [K-MEANS](#k-means)
		- [DBSCAN](#dbscan)
	- [Analysing](#analysing)

- - -

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
