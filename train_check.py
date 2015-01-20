"""Trains a neural network on a variety of test data and computes the cost"""

from scipy.io import loadmat
import numpy as np

import soldier
import tailor

print("Training on subset of MNIST Handwritten Digits dataset")

# Load data
print("Loading data...")
data = loadmat("./data/test/ex4data1.mat")
training_data = np.array(data["X"])
training_labels = np.array(data["y"])

# Reformat training labels
def format_label(label):
    new_label = np.zeros((10,))
    new_label[label[0]-1] = 1
    return new_label
training_labels = np.array(list(map(format_label, training_labels)))

# Instantiate network
print("Instantiating neural network...")
input_layer_size = training_data.shape[1]
hidden_layer_size = 25
num_labels = 10
reg = 0
nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size, num_labels,
                           reg,
                           soldier.sigmoid, soldier.sigmoid_gradient,
                           soldier.sigmoid, soldier.sigmoid_gradient)

print("Training neural network...")
np.seterr(over='raise')
nn.train(training_data, training_labels)
print(" Done")
