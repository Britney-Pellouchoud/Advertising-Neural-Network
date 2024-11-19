import Perceptron as pc
import Layer as lr
import numpy as np

data = np.array([[0.90, 0.87],
                 [1.81, 1.02],
                 [1.31, 0.75],
                 [2.36, 1.60],
                 [2.48, 1.14],
                 [2.17, 2.08],
                 [0.41, 1.87],
                 [2.85, 2.91],
                 [2.45, 0.52],
                 [1.05, 1.93],
                 [2.54, 2.97],
                 [2.32, 1.73],
                 [0.07, 0.09],
                 [1.86, 1.31],
                 [1.32, 1.96],
                 [1.45, 2.19],
                 [0.94, 0.34],
                 [0.28, 0.71],
                 [1.75, 2.21],
                 [2.49, 1.52]])

taca = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])

X_train = data[::2]
y_train = taca[::2]
X_test = data[1::2]
y_test = taca[1::2]

def online_once(outLayer, inputs, targetOutputs, eta):
    for i, inp in enumerate(inputs):
        outLayer = lr.ffbp_epoch(outLayer, inp, targetOutputs[i], eta)
    return outLayer

#initial weights are completely arbitrary
one = pc.Perceptron([0.4, 0.6])
two = pc.Perceptron([0.6, 0.4])
three = pc.Perceptron([0.5, 0.5])

#building the network
hidden = lr.Layer([one, two])
output = lr.Layer([three], hidden)
hidden.follow = output

eta = 0.5

#training
for i in range(30):
    out = online_once(output, X_train, y_train, eta)
    
out.previous.perc[0].print()
out.previous.perc[1].print()
out.perc[0].print()

#testing
for i, inp in enumerate(X_test):
    print(lr.feed_forward(out, inp), y_test[i])