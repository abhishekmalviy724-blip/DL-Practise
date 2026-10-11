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
__________________________________________
import numpy as np
import pandas as pd
a = pd.read_csv("healthcare_dataset.csv")
X = a[["Age","Billing Amount", "Room Number"]]
y = np.where(a["Test Results"]=="Normal",0,1)
w1 = np.array([
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4]
])
b1 = np.array([0.0,0.0,0.0,0.0])
w2 = np.array([
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4]
])
b2 = np.array([0.0,0.0,0.0,0.0])
Z1 = X @ w1 + b1
A1 = np.maximum(0,Z1)
Z2 = A1 @ w2 + b2
A2 = 1/(1+np.exp(-Z2))
pred = np.where(A2)
print(pred)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# ReLU Neuron with ...
import numpy as np
import pandas as pd
a=pd.read_csv("healthcare_dataset.csv")
X=np.array(a[["Age","Billing Amount","Room Number"]])
y=np.where(a["Test Results"]=="Normal",1,0)
w=np.array([0.0,0.0,0.0])
b=0.1
z = X @ w + b
print(1/(1+np.exp(-z)))
print(np.maximum(0,z))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________import numpy as np
from sklearn.metrics import mean_squared_error


X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]])
w = 0.5
b = 1.0
lr = 0.1
gr = 0.2
N = len(X)
z = X * w + b
print(z)
pred=1/(1+np.exp(-z))
print(pred)
print(mean_squared_error(y,z))
dw = (2/N)*np.max((z-y)*X)
db = (2/N)*np.sum(z-y)
print(dw)
print(db)
print(w - lr * dw)
print(b - lr * db)
print(pred*(1-pred))
print(pred-y)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

a = pd.read_csv("healthcare_dataset.csv")

X = a[["Age", "Billing Amount", "Room Number"]].values
y = np.where(a["Test Results"] == "Normal", 1, 0).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

np.random.seed(42)

W1 = np.random.randn(3, 2) * 0.1
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 2) * 0.1
b2 = np.zeros((1, 2))

W3 = np.random.randn(2, 1) * 0.1
b3 = np.zeros((1, 1))

lr = 0.01
epochs = 100

for epoch in range(epochs):

    z1 = X @ W1 + b1
    A1 = np.maximum(0, z1)

    z2 = A1 @ W2 + b2
    A2 = np.maximum(0, z2)

    z3 = A2 @ W3 + b3
    A3 = 1 / (1 + np.exp(-np.clip(z3, -500, 500)))

    loss = 0.5 * np.mean((A3 - y) ** 2)

    dA3 = (A3 - y) / len(X)
    dZ3 = dA3 * A3 * (1 - A3)

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W3 = W3 - lr * dW3
    b3 = b3 - lr * db3

    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1

    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss)

predictions = (A3 >= 0.5).astype(int)

print("Final Loss:", loss)
print("First 10 Predictions:")
print(predictions[:10].flatten())
print("First 10 Actual Values:")
print(y[:10].flatten())
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
