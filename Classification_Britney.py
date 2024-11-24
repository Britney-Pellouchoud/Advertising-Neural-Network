import numpy as np
import Perceptron as pc
import Layer as lr
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def online_once(outLayer, inputs, targetOutputs, eta):
    for i, inp in enumerate(inputs):
        outLayer = lr.ffbp_epoch(outLayer, inp, targetOutputs[i], eta)
    return outLayer


# Load the data
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

# Initialize perceptron
hidden = lr.Layer([pc.Perceptron([0.4, 0.6]), pc.Perceptron([0.6, 0.4])])
output = lr.Layer([pc.Perceptron([0.5, 0.5])], hidden)
hidden.follow = output

# Training the network
eta = 0.5
for epoch in range(30):
    output = online_once(output, X_train, y_train, eta)

# Optimize threshold
thresholds = np.linspace(0, 1, 100)
best_threshold = 0
best_accuracy = 0
predictions = [lr.feed_forward(output, x) for x in X_test]

for thresh in thresholds:
    preds = [1 if p >= thresh else 0 for p in predictions]
    accuracy = np.mean([pred == true for pred, true in zip(preds, y_test)])
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = thresh

print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")

fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

hidden = lr.Layer([pc.Perceptron([0.4, 0.6]), pc.Perceptron([0.6, 0.4]), pc.Perceptron([0.7, 0.3])])
output = lr.Layer([pc.Perceptron([0.5, 0.5])], hidden)
hidden.follow = output

