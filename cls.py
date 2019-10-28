# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:18:25 2019

@author: sb00747428
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
# reading the big mart sales training data

from sklearn.preprocessing import MinMaxScaler

df =pd.read_excel('data.xlsx')


scaler = MinMaxScaler(feature_range=(0, 1))
y = df.loc[:,['Interviewed']]
y = y*1

kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(x)
x= x.drop(columns='labels')

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))