# -*- coding: utf-8 -*-
"""
Spyder Editor
Xtrain, Xtest, Ytrain, Ytest= train_test_split()

This is a temporary script file.
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

df =pd.read_excel('data.xlsx')

for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat}".format(col_name=col_name, unique_cat= unique_cat))
        
        
        
        
#df = df.drop(columns =)

todummy = ['Gender','Job_Type','Race']

def dummy_data(df, todummy):
    for x in todummy:
        dummies = pd.get_dummies(df[x], prefix = x, dummy_na = False)
        df= df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df
    
new_data = dummy_data(df, todummy)



label =[]
for row in new_data['Interviewed']:
    if row == 'True':
        label.append(0)
    else:
        label.append(1)

new_data['label']= label

x = new_data.drop(columns = 'Name')
x = x.drop(columns = 'Interviewed')
x = x.drop(columns = 'Age')
y = new_data.loc[:,['label']]


x.isnull().sum().sort_values(ascending=False).head()

#imp = Imputer(missing_values="NaN", strategy = 'median', axis=0)
#imp.fit(x)
#x1 = pd.DataFrame(data=imp.transform(x), columns=x.columns)
#x1.isnull().sum().sort_values(ascending=False).head()

"""x1 = x1.loc[:,['match_id', 'location_x', 'location_y', 'remaining_sec', 'remaining_min.1', 'distance_of_shot.1', 'power_of_shot.1','remaining_sec.1', 'distance_of_shot']]"""
rnd = 1
test_size = 0.40
Xtrain,Xtest, Ytrain, Ytest =train_test_split(x,y, test_size=test_size, random_state = rnd)

training_model = XGBClassifier(objective = 'binary:logistic', max_depth=4, n_estimators=10000)
training_model.fit(Xtrain, Ytrain)
y_pred = training_model.predict(Xtest)
mae= mean_absolute_error(Ytest, y_pred)

score = 1/ (1+mae)
accuracy = training_model.score(Xtest, Ytest)
y_pred = training_model.predict_proba(Xtest)
predict = training_model.predict(Xtest)
y_predicted = training_model.predict(Xtest)
predictions = [round(value) for value in y_predicted]

accuracy = accuracy_score(Ytest, predictions)
xgb.plot_importance(training_model)
plt.rcParams['figure.figsize']= [20,20]
plt.show()
