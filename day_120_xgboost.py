import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, r2_score

print('version',xgb.__version__)

df=pd.read_csv('datasets/heart-disease.csv')
x=df.iloc[:,:-1]
y=df.iloc[:,-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

cls = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1 # Use all available cores
)
cls.fit(x_train,y_train)
y_pred=cls.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
f1=f1_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f1)
print(recall)
print(precision)