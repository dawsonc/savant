"""Trains a neural network on a variety of test data and computes the cost"""

from scipy.io import loadmat
import numpy as np

# Needed to access top-level stuff
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import soldier

print("Training on subset of MNIST Handwritten Digits dataset")

# Load data
print("Loading data...")
data_dict = loadmat("./data/test_data/ex4data1.mat")
data = np.array(data_dict["X"])
labels = np.array(data_dict["y"])

# Reformat training labels
def format_label(label):
    new_label = np.zeros((10,))
    new_label[label[0]-1] = 1
    return new_label
labels = np.array(list(map(format_label, labels)))

# Split up data into training & test data
# Shuffle the data first (some hackish-y stuff here)
data = np.array(list(zip(data, labels)))
np.random.shuffle(data)
split = round(data.shape[0] * 0.60)
training = data[:split]
test = data[split:]

training_data = np.array(list(map(lambda tup: tup[0], training)))
training_labels = np.array(list(map(lambda tup: tup[1], training)))

test_data = np.array(list(map(lambda tup: tup[0], test)))
test_labels = np.array(list(map(lambda tup: tup[1], test)))

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

# Calculate training error
predictions, _ = nn.predict(training_data)
training_error = np.mean(np.double(predictions == training_labels)) * 100
print("Training error: ", training_error)

# Calculate test error
predictions, _ = nn.predict(test_data)
test_error = np.mean(np.double(predictions == test_labels)) * 100
print("Test error: ", test_error)
