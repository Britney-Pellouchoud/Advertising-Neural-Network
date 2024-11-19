import Perceptron as pc

class Layer(object):
    def __init__(self, perceptrons, previous=None, following=None):
        self.perc = perceptrons
        self.previous = previous
        self.following = following
        
        
def feed_forward(out, inputs):
    hidden = out.previous.perc
    hiddenAct = []
    for perceptron in hidden:
        hiddenAct.append(perceptron.calc_activation(inputs))
    
    return out.perc[0].calc_activation(hiddenAct)

    
def ffbp_epoch(out, inputs, target, eta, train_bias=False):
    #FEED FORWARD
    hidden = out.previous.perc
    hiddenAct = []
    for j, perceptron in enumerate(hidden):
        hiddenAct.append(perceptron.calc_activation(inputs))
    
    y = out.perc[0].calc_activation(hiddenAct)
    error = target - y
    delta_k = error * (1 - y) * y
    
    #BACK PROPAGATION
    #k is index of output layer (only 1 in this case)
    newWeights_k = []
    for i, weight in enumerate(out.perc[0].weight):
        newWeights_k.append(weight + eta * delta_k * hiddenAct[i])
    newBias_k = out.perc[0].bias + eta * delta_k
    #BUT DON'T UPDATE YET!
    
    #j is index of hidden layer
    delta_j = []
    newWeights_j = []
    for j, perceptron in enumerate(hidden):
        sum_dkw = delta_k * out.perc[0].weight[j] #This sum is for all of k. Right now it's only 1, but if I had multiple outputs, it would be delta * weight of each of them.
        delta_j.append((1 - hiddenAct[j]) * hiddenAct[j] * sum_dkw)
        
        #i is the index of input
        newWeights_j.append([])
        for i, weight in enumerate(perceptron.weight):
            newWeights_j[j].append(weight + eta * delta_j[j] * inputs[i])
        
        hidden[j].weight = newWeights_j[j]
        if train_bias:
            hidden[j].bias = hidden[j].bias + eta * delta_j[j]
        #print("hidden ", j, " output: ", hidden[j].calc_activation(inputs))
    
    out.perc[0].weight = newWeights_k
    if train_bias:
        out.perc[0].bias = newBias_k
    
    return out

#NOTE TO SELF: in Perceptron, delta uses -error, then weight - eta * delta * input
#in ffbp_epoch, delta_k is calculated with +error, so then weight + eta * detla_k * input
#I think I did that because that's how it was in the slides. If I'm feeling particularly
#diligent one day maybe I'll reconcile it.

"""
#test stuff
inputs = [1, 2]
one = pc.Perceptron([0.3, 0.3])
two = pc.Perceptron([0.3, 0.3])
three = pc.Perceptron([0.8, 0.8])

hidden = Layer([one, two])
output = Layer([three], hidden)

afterOutput = ffbp_epoch(output, inputs, 0.7, 1, train_bias=True)
print(afterOutput.previous.perc[0].weight)
print(afterOutput.previous.perc[0].bias)
print(afterOutput.previous.perc[1].weight)
print(afterOutput.previous.perc[1].bias)
print(afterOutput.perc[0].weight)
print(afterOutput.perc[0].bias)
"""

"""
inputs = [1,3]
one = pc.Perceptron([0.8, 0.1])
two = pc.Perceptron([0.5, 0.2])
hidden = Layer([one, two])
oneOut = one.calc_activation(inputs)
twoOut = two.calc_activation(inputs)

print("Q1: ", oneOut)
print("Q2: ", twoOut)

three = pc.Perceptron([0.2, 0.7])
output = Layer([three], hidden)
hidden.following = output
threeOut = three.calc_activation([oneOut, twoOut])
print("Q3: ", threeOut)

after_epoch = ffbp_epoch(output, inputs, 0.95, 0.1)
oneOut_after = after_epoch.previous.perc[0].calc_activation(inputs)
twoOut_after = after_epoch.previous.perc[1].calc_activation(inputs)
threeOut_after = after_epoch.perc[0].calc_activation([oneOut_after, twoOut_after])

print("output node")
after_epoch.perc[0].print()
print("activation: ", threeOut_after)
print("node 1")
after_epoch.previous.perc[0].print()
print("activation: ", oneOut_after)
print("onde 2")
after_epoch.previous.perc[1].print()
print("activation: ", twoOut_after)
"""











