import numpy as np
from scipy import linalg
import soldier


def convert_labels(label, num_labels):
    label_vec = np.zeros((1, num_labels))
    label_vec[0, label - 1] = 1
    return label_vec


def compute_numerical_gradient(costf, nn_params):
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    e = 1e-4
    for p in range(nn_params.size):
        perturb[p] = e
        loss1 = costf(nn_params - perturb)[0]
        loss2 = costf(nn_params + perturb)[0]
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad

# Network parameters
reg = 0
input_layer_size = 3
hidden_layer_size = 5
num_labels = 3
m = 5

# Generate pseudo-random test data
theta1 = soldier.NeuralNetwork.random_initial_weights(input_layer_size,
                                                      hidden_layer_size)
theta2 = soldier.NeuralNetwork.random_initial_weights(hidden_layer_size,
                                                      num_labels)
# Reuse random_initial_weights to generate X
X = soldier.NeuralNetwork.random_initial_weights(input_layer_size - 1, m)

# Generate labels
y = 1 + np.linspace(1, m, m) % num_labels
Y = [convert_labels(label, num_labels) for label in y]

# Unroll parameters
nn_params = np.concatenate((theta1.ravel(), theta2.ravel()))

# Create network instance
nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size, num_labels,
                           reg,
                           soldier.sigmoid, soldier.sigmoid_gradient,
                           soldier.sigmoid, soldier.sigmoid_gradient)

costf = lambda params: nn.cost(X, Y, params)

cost, gradient = costf(nn_params)
numgrad = compute_numerical_gradient(costf, nn_params)

print("AnalyticalGrad   NumericalGrad")
for i in range(numgrad.size):
    print("%s  %s" % (gradient[i], numgrad[i]))
diff = linalg.norm(numgrad - gradient) / linalg.norm(numgrad + gradient)
print("Relative difference between analytical and numerical gradients: %s" %
      diff)
