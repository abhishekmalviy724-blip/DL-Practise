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
_____________________________________________________________________________________________________
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
a=pd.read_csv("healthcare_dataset.csv")
X=a[["Age", "Billing Amount", "Room Number"]]
y=np.where(a["Test Results"]=="Normal",0,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
w = np.array([0.0,0.0,0.0])
b = 0
learning_rate = 0.1
for i in range(5):
    for j in range(len(X_train)):
        z = X_train.iloc[j].values @ w + b
        pred = 1 if z>=0 else 0
        error = y_train[j] - pred
        w = w + learning_rate * error * X_train.iloc[j].values
        b = b +learning_rate * error 
print(w)
print(b)

z = X_test @ w + b
test_pred = np.where(z>=0,1,0)
correct = np.sum(y_test==test_pred)
total = len(y_test)
print(correct/total)

correc = np.sum(y_test==test_pred)
wrong = np.sum(y_test != test_pred)
accu = correc/len(y_test)*100
print(correc)
print(wrong)
print(accu)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# forward propagation
import numpy as np
X = np.array([2,3])
w1 = np.array([
    [0.5,0.4],
    [0.2,0.3]
])
b1 = np.array([1,-1])
w2 = np.array([0.6,0.9])
b2 = -1
Z1 = X @ w1 +b1
A1 = np.maximum(0,Z1)
Z2 = A1 @ w2 + b2
A2 = 1/(1+np.exp(-Z2))
pred1 = np.where(A1>=0,1,0)
pred2 = np.where(A2>=0.5,1,0)
print(pred1)
print(pred2)
print(Z1)
print(Z2)
print(A1)
print(A2)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
