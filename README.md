# DL-Practise 
# Preceptron
import numpy as np
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])
w = np.array([0.0,0.0])
b = -1
learning_rate = 0.1
for i in range(10):
    for j in range(len(X)):
        z = X[j] @ w + b
        pred = 1 if z>=0 else 0
        error = y[j]-pred
        w = w + learning_rate * error * X[j]
        b = b + learning_rate * error 
print(w)
print(b)
_______________________________________________________
import numpy as np
import pandas as pd
a=pd.read_csv("healthcare_dataset.csv")
X=a[["Age", "Billing Amount", "Room Number"]]
y=np.where(a["Test Results"]=="Normal",0,1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

w = np.array([0.0,0.0,0.0])
b = -1

learning_rate = 0.1

for i in range(20):
    for j in range(len(X_train)):
        z = X_train.iloc[j].values @ w + b
        pred = 1 if z>=0 else 0 
        error = y_train[j] - pred
        w = w + learning_rate * error * X_train.iloc[j].values
        b = b + learning_rate * error 
print(w)
print(b)
_______________________________________________________________________________________________________________________________________________________________________________________________________________________
