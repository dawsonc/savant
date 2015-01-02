"""Module for running forward- and back-propagation on the ANN"""

import scipy
from scipy import linalg
from scipy import optimize
import numpy as np


class NeuralNetwork(object):

    """A neural network with one hidden layer"""
    @staticmethod
    def random_initial_weights(incoming_connections, outgoing_connections):
        """Return the random intial weights for a layer with the given number \
of connections

        Arguments:
            incoming_connections -- the number of connections going into each
                                     neuron in the layer
            outgoing_connections -- the number of neurons in this layer
        """
        epsilon = np.sqrt(
            6) / np.sqrt(incoming_connections + outgoing_connections)
        # +1 accounts for bias node
        weights = np.random.rand(
            outgoing_connections, incoming_connections + 1)
        weights = weights * (2 * epsilon) - epsilon
        return weights

    def __init__(self, input_layer_size, hidden_layer_size, num_outputs, reg,
                 activation_function, activation_function_gradient,
                 output_activation_func, output_activation_func_gradient):
        """Initializes the neural network with the given parameters

        Arguments:
            input_layer_size             -- the # of nodes in the input layer
            hidden_layer_size            -- the # of nodes in the hidden layer
            num_outputs                  -- the # of nodes in the output layer
            reg                          -- the regularization parameter
            activation_function          -- the activation fn for the neurons
            activation_function_gradient -- the neuron activation fn gradient
            output_activation_func       -- the activation fn for output \
neurons
            output_activation_func_gradient
                            -- the gradient of the output neuron activiation fn
        """
        self.i_size = input_layer_size
        self.h_size = hidden_layer_size
        self.o_size = num_outputs

        self.theta1 = NeuralNetwork.random_initial_weights(
            self.i_size, self.h_size)
        self.theta2 = NeuralNetwork.random_initial_weights(
            self.h_size, self.o_size)

        self.reg = reg

        self.activation_function = activation_function
        self.activation_function_gradient = activation_function_gradient
        self.output_activation_func = output_activation_func
        self.output_activation_func_gradient = output_activation_func_gradient

    def train(self, indicators, labels):
        pass

    def predict(self, indicators):
        pass

    def cost(self, indicators, labels, nn_params):
        """Computes the cost and gradient for the network using \
the given parameters

        Arguments (these should all be numpy.ndarray):
            indicators -- the feature matrix for which to compute the \
cost/gradient
            labels     -- the labels for each of the examples in indicators
                            Note: indicators and labels must have the same \
number of rows
            nn_params  -- the "unrolled" version of theta1 and theta2
                           . should be formed by concatenating theta1.ravel() \
and theta2.ravel()

        Returns the tuple (cost, gradient), where:
            cost     -- The root-mean-square prediction error of the network
            gradient -- The "unrolled" concatenation of the gradients for \
theta1 and theta2
        """
        # First get theta1 and theta2 back into fighting shape
        theta1 = nn_params[0:self.theta1.size]
        theta1.shape = self.theta1.shape
        theta2 = nn_params[self.theta1.size:]
        theta2.shape = self.theta2.shape

        # Then define some useful values
        num_examples = indicators.shape[0]

        # Now start the music
        # Part Ia: Feedforward

        # Add bias value to inputs
        bias = np.ones((num_examples, 1))
        indicators = np.concatenate((bias, indicators), axis=1)

        # Compute z values for the hidden layer (layer 2) neurons
        # . z_hidden is a num_examples x self.h_size array
        z_hidden = indicators.dot(theta1.T)
        # Compute activation values for layer 2 neurons
        # . activation_hidden is a num_examples x self.h_size array
        activation_hidden = self.activation_function(z_hidden)
        # Add bias value to hidden layer
        activation_hidden = np.concatenate((bias, activation_hidden), axis=1)

        # Compute z values for the output layer neurons
        # . z_output is a num_examples x self.o_size array
        z_output = activation_hidden.dot(theta2.T)
        # Compute activation values for output neurons (the hypotheses matrix)
        # . hypotheses is a num_examples x self.o_size array
        hypotheses = self.output_activation_func(z_output)

        # Part Ib: Cost Function

        # Iterate over the examples to sum the cost
        cost = 0
        for i in range(num_examples):
            label = labels[i]
            hypothesis = hypotheses[i]

            cost += np.sum((-1 * label) * np.log(hypothesis) -
                           (1 - label) * np.log(1 - hypothesis))
        cost *= 1 / num_examples

        # Add regularization
        cost += self.reg / (2 * num_examples) * \
            (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

        # Part II: Backpropogation
        # Vectorization will come later ;)
        D1 = np.zeros(theta1.shape)
        D2 = np.zeros(theta2.shape)
        for i in range(num_examples):
            # Compute output error
            delta3 = (hypotheses[i] - labels[i]).T

            # Backprop to hidden layer
            # . removing error for bias unit as we go
            z_hidden_col = z_hidden[i]
            z_hidden_col.shape = (z_hidden_col.shape[0], 1)
            delta2 = (theta2.T.dot(delta3))[1:] * \
                self.activation_function_gradient(z_hidden_col)

            # Accumulate the little deltas into the big Deltas (crystal clear,
            # right?)
            indicator = indicators[i]
            indicator.shape = (1, indicator.shape[0])
            D1 += delta2.dot(indicator)

            activation = activation_hidden[i]
            activation.shape = (1, activation.shape[0])
            D2 += delta3.dot(activation)
        # Scale and regularize
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)
        # Don't regularize bias term
        theta1_grad[:, 0] = 1 / num_examples * D1[:, 0]
        theta2_grad[:, 0] = 1 / num_examples * D2[:, 0]
        # Regularize everything else
        theta1_grad[:, 1:] = 1 / num_examples * D1[:, 1:] + \
            self.reg / num_examples * theta1[:, 1:]
        theta2_grad[:, 1:] = 1 / num_examples * D2[:, 1:] + \
            self.reg / num_examples * theta2[:, 1:]

        # Unroll gradients
        gradients = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))
        return (cost, gradients)


def sigmoid(z):
    """Computes the output of the sigmoid function for the input z"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_gradient(z):
    """Computes the gradient of the sigmoid function evaluated at z"""
    return sigmoid(z) * (1 - sigmoid(z))