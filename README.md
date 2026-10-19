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
_____________________________________________________________
import numpy as np
X=np.array([[2.0,3.0]])
w1 = np.array([
    [0.1,0.2],
    [0.3,0.4]
])
b1 = np.array([[0.1,0.2]])
w2 = np.array([
    [0.5],
    [0.6]
])
b2 = np.array([
    [0.2]
])
z1 = X @ w1 + b1
a1 = np.maximum(0,z1)
z2 = a1[:,None] @ w2 + b2
a2 = 1/(1+np.exp(-z2))

import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = np.array([1.0, 0.0, 1.0, 1.0])
y_pred = np.array([0.8, 0.3, 0.6, 0.9])
loss = -np.mean(y_true * np.log(y_pred) + (1-y_true)* np.log(1-y_pred))
loss

import numpy as np
X = np.array([[2.0], [3.0]])
y_true = np.array([[1.0], [0.0]])
y_pred = np.array([[0.8], [0.3]])
err = y_pred - y_true
gr = X.T @ err / len(y_true)
print(np.mean(err))
print(err)
print(gr)

import numpy as np

w = np.array([[0.5]])
b = 0.1

dw = np.array([[-0.15]])
db = -0.05

learning_rate = 0.1
w = w - learning_rate * dw
b = b - learning_rate * db
print(w)
print(b)


import numpy as np

X = np.array([[1.0], [2.0]])
y = np.array([[1.0], [0.0]])

w = np.array([[0.5]])
b = 0.1

learning_rate = 0.1
z = X @ w + b
a1 = 1/(1+np.exp(-z))
err = a1 - y
dw = X.T @ err / len(y)
db = np.mean(err)
w = w + learning_rate * dw
b = b + learning_rate * db
z
a1
err
dw
db
w
b


import numpy as np
X = np.array([[1.0], [2.0]])
y = np.array([[1.0], [0.0]])

w = np.array([[0.5]])
b = 0.1
learning_rate = 0.1
for i in range(100):
    z = X @ w + b
    pred = 1/(1+np.exp(-z))
    err = pred - y
    dw = X.T @ err/len(y)
    db = np.mean(err)
    w = w - learning_rate * dw
    b = b - learning_rate * db
print(w)
print(b)
print(err)
print(pred)
print(z)
______________________________________________

import numpy as np
x = 2.0
y = 1.0
w = 0.5
b = 0.1
w2 = 0.5
b2 = 0.1
learning_rate = 0.1
z1 = x * w + b
a1 = 1/(1+np.exp(-z1))
z2 = a1 * w2 + b2
a2 = 1/(1+np.exp(-z2))
err = a2 - y
dw = err * X
db = err 
dz1 = err * w2 * a1 * (1-a1)
dw1 = dz1 * X
dw2 = dz1
w = w -  learning_rate * dw1
b = b - learning_rate * dw2
print(w)
print(b)

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import numpy as np
z = np.array([2.0, 1.0, 0.1])
a = np.exp(z)
su = np.sum(a)
print(a/su)
______________________________
import numpy as np

z = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 2.0],
    [0.5, 0.2, 1.5]
])
a = np.exp(z)
b = np.sum(a,axis=1,keepdims=True)
print(a/b)
_____________________________________
import numpy as np

z = np.array([
    [1000.0, 1001.0, 1002.0],
    [500.0, 502.0, 501.0]
])
b = np.max(z,axis=1,keepdims=True)
sub = b-z
exp = np.exp(sub)
su = np.sum(exp,axis=1,keepdims=True)
print(exp/su)
_________________________________________
z = np.array([3.0, 1.0, 2.0])
a = np.max(z)-z
b = np.exp(a)
c = np.exp(z - np.max(z))
d = c/np.sum(c)
print(np.max(d))
________________________________________
z = np.array([3.0, 1.0, 2.0])
a = np.exp(z - np.max(z))
b = a/np.sum(a)
c = np.sum(b)
c
____________________________________
z = np.array([3.0, 1.0, 2.0])
a = np.exp(z - np.max(z))
b = a/np.sum(a)
print(np.argmax(b))
________________________________
import numpy as np
z = np.array([
    [2.0, 5.0, 1.0],
    [7.0, 2.0, 3.0],
    [1.0, 4.0, 6.0]
])
a = z - np.max(z)
b = np.exp(a)
c = b/np.sum(b,axis=1,keepdims=True)
print(c)
______________________________________
import numpy as np
z = np.array([
    [2.0, 5.0, 1.0],
    [7.0, 2.0, 3.0],
    [1.0, 4.0, 6.0]
])
a = z - np.max(z, axis=1, keepdims=True)
b = np.exp(a)
c = b / np.sum(b, axis=1, keepdims=True)
d=np.argmax(c,axis=1,keepdims=True)
print(np.sum(d,axis=1,keepdims=True))
________________________________________
import numpy as np

z = np.array([
    [2.0, 5.0, 1.0],
    [7.0, 2.0, 3.0],
    [1.0, 4.0, 6.0]
])
a = z - np.max(z)
b = np.exp(a)
c = b/np.sum(b,axis = 1, keepdims=True)
print(np.argmax(c,axis = 1, keepdims=True))
________________________________________________
import numpy as np
X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [1.0, 1.0]
])

W = np.array([
    [0.2, 0.5, 0.1],
    [0.4, 0.3, 0.6]
])
b = np.array([[0.1, 0.2, 0.1]])
z = X @ W + b
_____________________________________________
c = np.max(z,axis=1)
d = np.exp(c)
print(d/np.sum(d))
print(c)
______________________________________
c = z - np.max(z)
d = np.exp(c)
e = np.exp(d - np.max(d))
f = e/np.sum(e)
print(f)
print(np.sum(f,axis=1))
print(np.argmax(f,axis=1,keepdims=True))
_______________________________________________
import numpy as np
y_true = 1
y_pred = 0.8
loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
loss
_________________________________________________________________________
y_true = 0
y_pred = 0.8
print(-(y_true * np.log(y_pred)+ (1-y_true)* np.log(1-y_pred)))
____________________________________________________________________
import numpy as np
y_true = 2
y_pred = np.array([0.1, 0.2, 0.7])
print(-(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))
print(-np.log(y_pred[y_true]))
___________________________________________________________________
import numpy as np
y_true = 0
y_pred = np.array([0.6, 0.3, 0.1])
print(-(y_true)*np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
print(-np.log(y_pred[y_true]))
______________________________________________________________________
y_true = 0
y_pred = 0.3
print(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred)))
_______________________________________________________________________
import numpy as np
y_true = 3
y_pred = np.array([0.1, 0.2, 0.3, 0.4])
print(-np.log(y_pred[y_true]))
_____________________________________________
import numpy as np
y_true = 3
y_pred = np.array([0.1, 0.6, 0.2, 0.1])
print(-(np.log(y_pred[y_true])))
________________________________________________________
import numpy as np 
x = 2.0
y = 0.0

w1 = 0.5
b1 = 0.1

w2 = 0.8
b2 = 0.2

lr = 0.1

z1 = x * w1 + b1
a1 = 1/(1+np.exp(-z1))
z2 = a1 * w2 + b2
a2 = 1/(1+np.exp(-z1))
err = a2 - y
dz2 = 2 * err * a2 * (1 - a1)
dw2 = dz2 * a1
db2 = dz2
dz1 = dz2 * w2 * a2 * (1 - a2)
dw1 = dz1 * x
db1 = dz1
w1 = w1 - lr * dw1
w2 = w2 - lr * dw2
b1 = b1 - lr * db1
b2 = b2 - lr * db2
old_loss = (a2 - y)**2

a = x * w1 + b1
b = 1/(1+np.exp(-a))
c = b * w2 + b2
d = 1/(1+np.exp(-c))
n = d - y **2
n
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])

w = np.array([0.2, 0.3])
b = -0.4
learning_rate = 0.1

for i in range(10):
    for j in range(len(X)):
        z = np.dot(X[j],w)+b
        pred = 1 if z>=0 else 0 
        err = y[j] - pred
        w = w + learning_rate * err * X[j]
        b = b + learning_rate * err
        
y_pred = []
for k in range(len(X)):
    z = np.dot(X[k],w)+b
    pred1 = 1 if z>= 0 else 0
    y_pred.append(pred1)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import numpy as np

X = np.array([1.0, 2.0, 3.0])
Y = np.array([1.0, 0.0, 1.0])

w1 = 0.5
b1 = 0.1
w2 = 0.8
b2 = 0.2
learning_rate = 0.1

z1 = X * w1 + b1
a1 = 1 / (1 + np.exp(-z1))

z2 = a1 * w2 + b2
a2 = 1 / (1 + np.exp(-z2))

loss = -np.mean(Y * np.log(a2) + (1-Y) * np.log(1-a2))

dz2 = a2 - Y
dw2 = np.mean(dz2 * a1)
db2 = np.mean(dz2)
dz1 = dz2 * w2 * a1 * (1 - a1)
dw1 = np.mean(dz1 * X)
db1 = np.mean(dz1)
w1 = w1 - learning_rate * dw1
w2 = w2 - learning_rate * dw2
b1 = b1 - learning_rate * db1
b2 = b2 - learning_rate * db2
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import tensorflow as tf
a=pd.read_csv("healthcare_dataset.csv")
X=a.drop(columns=["Name", "Doctor", "Hospital", "Date of Admission", "Discharge Date"])
y=a["Test Results"]
le = LabelEncoder()
encoded = le.fit_transform(y)
b = X.copy()
b = pd.get_dummies(b,drop_first=True)
b.isnull().sum()
encoded.dtype

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(b,encoded,test_size=0.2,random_state=42)
st = StandardScaler()
y = st.fit_transform(X_train)
v = st.transform(X_test)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
print(len(a))
print(len(X_train)+len(X_test))
print(len(a)==len(X_train)+len(X_test))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

a=pd.read_csv("healthcare_dataset.csv")

X=a[["Age","Billing Amount","Room Number"]]
y=a["Test Results"]

X=X.fillna(X.median(numeric_only=True))
y=y.fillna(y.mode()[0])

le = LabelEncoder()
encoded = le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,encoded,test_size=0.2,random_state=42)

st = StandardScaler()
scaledX_train = st.fit_transform(X_train)
scaledX_test = st.transform(X_test)

model = Sequential([
    Input(shape=(3,)),
    Dense(16,activation="relu"),
    Dense(8,activation="relu"),
    Dense(3,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(scaledX_train,y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1)
test_loss,test_accuracy=model.evaluate(scaledX_test,y_test)
____________________________________________________________________________________________
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


a=pd.read_csv("healthcare_dataset.csv")
X=a[["Age", "Billing Amount", "Room Number"]]
y=a["Test Results"]
X=X.fillna(X.median())
y=y.fillna(y.mode()[0])

le = LabelEncoder()
labeld = le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,labeld,test_size=0.2,random_state=42)

st = StandardScaler()
scaled_train = st.fit_transform(X_train)
scaledy = st.transform(X_test)
num_classes=len(le.classes_)
model = Sequential([
    Input(shape=(3,)),
    Dense(32,activation="relu"),
    Dense(16,activation="relu"),
    Dense(8,activation="softmax"),
    Dense(num_classes, activation="softmax")
])
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    scaled_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
loss_t,accuracy=model.evaluate(scaledy,y_test)
pred = model.predict(scaledy)
print(pred[:5])
p=np.argmax(pred,axis=1)
inv = le.inverse_transform(p)
inv2 = le.inverse_transform(y_test)
compare = pd.DataFrame({
    "Actual":inv[:10],
    "pred":inv2[:10]
})

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import numpy as np

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [1.0, 6.0],
    [15.0, 2.0]
]) 

y = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
]) 

m = len(X)

W1 = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
])

b1 = np.array([[0.0, 0.0, 0.0, 0.0]])  

W2 = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

b2 = np.array([[0.0]])  
lr = 0.1
for i in range(10000):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))
    loss = -np.mean(y * np.log(a2 + 1e-15) + (1 - y) * np.log(1 - a2 + 1e-15))
    dz2 = a2 - y
    dw2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    relu_mask = np.where(z1 > 0, 1.0, 0.0)
    dz1 = (dz2 @ W2.T) * relu_mask
    dw1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    w1 = W1 - lr * dw1
    w2 = W2 - lr * dw2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2
    final_pred = X * W2 + b2
    final_loss = -np.mean((a2 - y)**2)

print(final_pred)
print(final_loss)
_____________________________________________________________________________________
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Softmax, Dropout, Flatten,Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist


a=pd.read_csv("healthcare_dataset.csv")
X=a[["Age","Billing Amount","Room Number"]]
y=a["Test Results"]
X=X.fillna(X.median())
y=y.fillna(y.mode()[0])

(X_train,y_train),(X_test,y_test)=mnist.load_data()
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train/255.0
X_test=X_test/255.0

model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_train, y_train, epochs=5)
test_loss,accuracy=model.evaluate(X_test,y_test)
pred = model.predict(X_test)
print(pred[:10])
print(np.argmax(model.predict(X_test),axis=1))
import matplotlib.pyplot as plt
i = 1
plt.imshow(X_test[i],cmap='gray')
plt.show()
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(
        filters=16,
        kernel_size=(5,5),
        activation="relu"
    )
])
model.summary()
_____________________________________________________

model1 = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(8, (3,3), activation="relu"),
    MaxPooling2D((2,2))
])
model.summary()
____________________________________________________
model2 = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
model.summary()
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
from  common_imports import *
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train = X_train/255
X_test = X_test/255

model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(16, (5,5),padding="valid",strides=1, activation="relu"),
    Conv2D(32, (3,3), padding="same",strides=2, activation="relu"),
    MaxPooling2D(pool_size=(2,2),strides=2),
])
model.summary()
__________________________________________________________________________
model1 = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), padding="same",dilation_rate=2,activation="relu")
])
model1.summary()
___________________________________________________________________________
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(16, (3,3), padding="same",dilation_rate=2, activation="relu"),
    Conv2D(32, (3,3), padding="valid", dilation_rate=1,activation="relu")
])
model.summary()
___________________________________________________________________________
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(8, (3,3), padding="same", activation="relu"),
    Conv2D(16, (1,1), padding="same", activation="relu"),
    Conv2D(32, (3,3), padding="same", activation="relu")
])
model.summary()

