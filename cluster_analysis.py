# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:01:56 2019

@author: sb00747428
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import sklearn.feature_selection
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib.font_manager
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
from sklearn.ensemble import IsolationForest

df =pd.read_excel('data.xlsx')

for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat}".format(col_name=col_name, unique_cat= unique_cat))
        


todummy = ['Gender','Job_Type','Race']

def dummy_data(df, todummy):
    for x in todummy:
        dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
        df= df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
    
new_data = dummy_data(df, todummy)

y = new_data.loc[:,['Interviewed']]
y = y*1
#y= y.values.tolist()


x = new_data.drop(columns = 'Name')
x = x.drop(columns = 'Interviewed')
x= x.drop(columns='id')
x = x.drop(columns = 'Age')
#x['labels']= y['Interviewed']

# by default the outlier fraction is 0.1 in generate data function 
outlier_fraction = 0.1

# store outliers and inliers in different numpy arrays
np.random.seed(1)
clf = IsolationForest(behaviour = 'new', n_estimators= 100, max_samples = 200, random_state =1 , contamination = 0.04)

preds = clf.fit_predict(x)


x['preds']= preds
x['labels']= y
#separate the two features and use it to plot the data 
new_x = x.loc[x['preds'] == -1]
fresh_x = x.loc[x['preds'] == 1]
# create a meshgrid 
xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))


y =[]


classifiers = {
     'Angle-based Outlier Detector (ABOD)'   : ABOD(contamination=outlier_fraction),
     'K Nearest Neighbors (KNN)' :  KNN(contamination=outlier_fraction)
}


xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
#set the figure size
plt.figure(figsize=(10, 10))

for i, (clf_name,clf) in enumerate(classifiers.items()) :
    # fit the dataset to the model
    clf.fit(x)
    xxx = x.to_numpy()
    # predict raw anomaly score
    scores_pred = clf.decision_function(xxx)*-1

    # prediction of a datapoint category outlier or inlier
    #x = x.reshape(1, -1)
    y_pred = clf.predict(xxx)

    # no of errors in prediction
    n_errors = (y_pred != y).sum()
    print('No of Errors : ',clf_name, n_errors)

    # rest of the code is to create the visualization

    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred,100 *outlier_fraction)

    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    subplot = plt.subplot(1, 2, i + 1)

    # fill blue colormap from minimum anomaly score to threshold value
    subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), threshold, 10),cmap=plt.cm.Blues_r)

    # draw red contour line where anomaly score is equal to threshold
    a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')

    # scatter plot of inliers with white dots
    b = subplot.scatter(x[:-n_outliers, 0], x[:-n_outliers, 1], c='white',s=20, edgecolor='k') 
    # scatter plot of outliers with black dots
    c = subplot.scatter(x[-n_outliers:, 0], x[-n_outliers:, 1], c='black',s=20, edgecolor='k')
    subplot.axis('tight')

    subplot.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')

    subplot.set_title(clf_name)
    subplot.set_xlim((-10, 10))
    subplot.set_ylim((-10, 10))
    plt.show() 
    
  fresh_x = pd.read_csv('fresh_x.csv')
accounting = fresh_x.loc[fresh_x['Job_Type_Accounting'] == 1]

female = accounting.loc[accounting['Gender_F']==1] 
male = accounting.loc[accounting['Gender_M']==1]
x1 = female.loc[:,['GPA']]
x2 = male.loc[:,['GPA']]
y1 = female.loc[:,['Qualification_type']]
y2 = male.loc[:,['Qualification_type']]

x1 = x1.values
x2 = x2.values
y1 = y1.values
y2=y2.values
r0 = 0.6
r = np.sqrt(x ** 2 + y ** 2)
N = 286
area = (20 * np.random.rand(N))**2
area1 = np.ma.masked_where(r < r0, area)
area2 = np.ma.masked_where(r >= r0, area)
plt.scatter('x1', 'y1',  c="red")
plt.scatter('x2', 'y2', c='blue')
plt.scatter(x1, y1, color='r')
plt.scatter(x2, y2, color='g')
plt.show()

plt.plot(x1, y1, 'red', x2, y2, 'bs')
plt.scatter('x1', '', c='c', s='d', data=data)

plt.show()