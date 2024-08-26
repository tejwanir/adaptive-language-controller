import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the neural network model
model = Sequential()

# Add the input layer (specify input_dim for the first layer)
model.add(Dense(10, input_dim=10, activation='linear'))  # 6 inputs to 4 hidden neurons with ReLU activation
model.add(Dense(10, input_dim=10, activation='relu'))  # 6 inputs to 4 hidden neurons with ReLU activation
model.add(Dense(10, input_dim=4, activation='linear'))  # 6 inputs to 4 hidden neurons with ReLU activation
model.add(Dense(4, input_dim=4, activation='relu'))  # 6 inputs to 4 hidden neurons with ReLU activation

# Add the output layer
model.add(Dense(4, activation='softmax'))  # 1 output neuron with Sigmoid activation (for binary classification)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




f = open("nndata.txt","r")
num = '0123456789.-e'
def cin():
    s = ""
    while True:
        nx = f.read(1)
        if nx in num:
            s += nx
            break
    while True:
        nx = f.read(1)
        if nx in num:
            s += nx
        else:
            break
    return s
    
X=[]
y=[]

x1=0
x2=0
y1=0
y2=0
z1=0
z2=0

p=[]

out = 1
for i in range(78):

    p = [x1,y1,z1,x2,y2,z2,0,0,0,0]
    p[out+5]=1
    x1 = float(cin())
    y1 = float(cin())
    z1 = float(cin())

    x2 = float(cin())
    y2 = float(cin())
    z2 = float(cin())
    out = cin()
    out = int(out)
    if i != 0:
        X.append(p)
        t = [0,0,0,0]
        t[out-1]=1
        y.append(t)


for i in range(77):
    print(X[i],y[i])



history = model.fit(np.array(X), np.array(y), verbose=1, epochs=100000)
predictions = model.predict(np.array(X))
for i in range(0,77):
    idx = 0
    for j in range(4):
        if predictions[i][idx] < predictions[i][j]:
            idx = j
    print(idx+1)


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

#model.save('nnmodel.keras')