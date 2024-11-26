import math

class Perceptron_FAR(object):
    def __init__(self, weights, bias=0):
        self.weight = weights
        self.bias = bias
        #self.activation
    
    #The activity of this perceptron is based on the difference between it's two inputs.
    #It will return a higher number if they are far apart.
    #It only handles 2 inputs. It's not very robust, but it serves my purpose.
    def calc_activity(self, inputs):
        return 1 - 1 / abs(inputs[0] * self.weight[0] - inputs[1] * self.weight[1])
    
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