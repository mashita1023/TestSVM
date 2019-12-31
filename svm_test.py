#%%
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

# sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#%%
df1 = pd.read_csv('DataSet_Social_Computer.csv')
# print(df1.head())

# 正規化　
#dfs = df1.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)
X = df1.iloc[:, :14401]
y = df1.iloc[:, 14401:14402]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# svm
print('\nSVM')
param_list = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [1, 2, 3, 4, 5]},
              {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4]}]

grid_search = GridSearchCV(SVC(), param_list, cv=5)
grid_search.fit(X_train, y_train)

# random forest
print('\nRandomForest')
param_list = {'n_estimators': [i for i in range(10, 100, 10)], 'criterion': ['gini', 'entropy'], 'max_depth': [i for i in range(1, 5, 1)], 'min_samples_split': [2, 4, 10, 12, 16], 'random_state': [3]}

grid_search_forest = GridSearchCV(RandomForestClassifier(), param_list, cv=5)
grid_search_forest.fit(X_train, y_train)

# result
print('SVM result')
print(grid_search.cv_results_)
print ('Test set score: {}'.format(grid_search.score(X_test, y_test)))
print ('Best parameters: {}'.format(grid_search.best_params_))
print ('Best cross-validation: {}'.format(grid_search.best_score_))
print('\nRandomForest result')
print ('Test set score: {}'.format(grid_search_forest.score(X_test, y_test)))
print ('Best parameters: {}'.format(grid_search_forest.best_params_))
print ('Best cross-validation: {}'.format(grid_search_forest.best_score_))
