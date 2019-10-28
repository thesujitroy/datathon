# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:34:24 2019

@author: sb00747428
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:47:50 2019

@author: sb00747428
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:24:43 2019

@author: sb00747428
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def dummy_data(df, todummy):
    for x in todummy:
        dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
        df= df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


def create_mlp(dim):
	# define our MLP network
    model = Sequential()
    model.add(Dense(80, input_dim=dim, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    	#model.add(Dense(4, activation="relu"))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
	# return our model
    return model






if __name__ == '__main__':
    df =pd.read_excel('data.xlsx')
    todummy = ['Gender','Job_Type','Race']
    new_data = dummy_data(df, todummy)

    
    x = new_data.drop(columns = 'Name')
    x = x.drop(columns = 'Interviewed')
    x= x.drop(columns='id')
    #x= x.drop(columns='labels')
    #x = x.drop(columns = 'Age')
   
    y = new_data.loc[:,['Interviewed']]
    y = y*1


    # train the model

    '''validation check'''
    Xtrain,Xtest, Ytrain, Ytest =train_test_split(x,y, test_size=0.20, random_state = 1)
    model = create_mlp(x.shape[1])
    opt = Adam(lr=1e-4, decay=1e-3 / 51)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    model.fit(Xtrain, Ytrain,validation_data=(Xtest, Ytest),epochs=100, batch_size=10)
    preds = model.predict(Xtest)
    predictions = [round(value) for value in preds]
    accuracy = accuracy_score(Ytest, predictions)
#    diff = preds - Ytest
#    percentDiff = (diff / Ytest) * 100
#    absPercentDiff = np.abs(percentDiff)
#    mean = np.mean(absPercentDiff)
#    std = np.std(absPercentDiff)
    '''                                      '''
#    print("[INFO] training model...")
#    #testY = test["price"] / maxPrice
#    model = create_mlp(trainX.shape[1], regress=True)
#    opt = Adam(lr=1e-4, decay=1e-3 / 50)
#    model.compile(loss="mean_squared_error", optimizer=opt)
#    model.fit(trainX, trainY,epochs=50, batch_size=10)
#    preds = model.predict(testX)
#    testX['Residual_Oxygen_(%)']=preds
#
#    result = pd.concat([traindf,testX], join = 'outer')
#    result.sort_index(inplace=True)
#    result['Residual_Oxygen_(%)'] = result['Residual_Oxygen_(%)']*maxoxy
#    df_miss1['Residual_Oxygen_(%)'] = result['Residual_Oxygen_(%)']
#    export_csv = df_miss1.to_csv (r'C:\Users\sb00747428\Downloads\pepsico challenge\predict_Residual_Oxygen.csv', index = None, header=True)
