import math


class Perceptron(object):
    def __init__(self, weights, bias=0):
        self.weight = weights
        self.bias = bias
        #self.activation
    
    def calc_activity(self, inputs):
        activity = self.bias
        for i, inp in enumerate(inputs):
            activity = activity + inp*self.weight[i]
        return activity
    
    def calc_activation(self, inputs):
        return 1 / (1 + math.exp(-self.calc_activity(inputs)))
    
    def print(self):
        print("weights: ", self.weight)
        print("bias:    ", self.bias)
        
    def train_once(self, inputs, target, eta, train_bias=False):
        newWeights = []
        activation = self.calc_activation(inputs)
        error = target - activation
        delta = -error * (1 - activation) * activation
        
        for i, weight in enumerate(self.weight):
            newWeights.append(weight - eta * delta * inputs[i])
            
        if train_bias:
            self.bias = self.bias - eta * delta
        self.weight = newWeights
        
        return self.calc_activation(inputs)
        

"""       
inp = [0.8, 0.9]
q1r = Perceptron([0.24, 0.88])
print(q1r.calc_activation(inp))
#print(q1r.activation)
for i in range(75):
    q1r.train_once(inp, 0.95, 5.0, train_bias=True)
print(q1r.weight, q1r.bias)
print(q1r.calc_activation(inp))
"""