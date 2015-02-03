"""Checks the neural network cost function using given thetas and data"""

from scipy.io import loadmat
import numpy as np

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import soldier

# Setup parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# Load data
data = loadmat("./data/test_data/ex4data1.mat")
training_data = np.array(data["X"])
training_labels = np.array(data["y"])

# Reformat training labels


def format_label(label):
    new_label = np.zeros((10,))
    new_label[label[0] - 1] = 1
    return new_label
training_labels = np.array(list(map(format_label, training_labels)))

# Load saved Neural Network params
params = loadmat("./data/test_data/ex4weights.mat")
theta1 = params['Theta1']
theta2 = params['Theta2']
nn_params = np.concatenate((theta1.ravel(), theta2.ravel()))

# First without regularization
reg = 0

print("Feedforward using Neural Network...")

nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size, num_labels,
                           reg,
                           soldier.sigmoid, soldier.sigmoid_gradient,
                           soldier.sigmoid, soldier.sigmoid_gradient)

print("Cost at parameters loaded from ex4weights w/o reg: %s" %
      nn.cost(training_data, training_labels, nn_params)[0])
print("This value should be about 0.287629")

# Now with regularization
reg = 1

print("Feedforward using saved Neural Network")

nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size, num_labels,
                           reg,
                           soldier.sigmoid, soldier.sigmoid_gradient,
                           soldier.sigmoid, soldier.sigmoid_gradient)

nn_params = np.concatenate((theta1.ravel(), theta2.ravel()))

print("Cost at parameters loaded from ex4weights w/ reg: %s" %
      nn.cost(training_data, training_labels, nn_params)[0])
print("This value should be about 0.383770")
