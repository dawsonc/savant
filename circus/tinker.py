"""Wrapper class for soldier's neural network implementation"""

import scipy.io
import numpy as np

from circus import soldier


class Tinker(object):

    """A class for retrieving data, training, & calculuation errors.

    Has multiple methods for training on different feature sets
    """

    def __init__(self, path_to_data):
        """Inits the object by loading the data from the .mat"""
        self.data_dict = scipy.io.loadmat(path_to_data)

    @staticmethod
    def error(predictions, labels):
        """Returns the % error of the neural network on the given examples"""
        pct_error = np.mean(np.double(np.equal(predictions, labels))) * 100

        return pct_error

    def baseline(self):
        """A baseline test that always "predicts" buy ([1, 0, 0])

        Returns a list of [training_error, cv_error, test_error"""
        # Load training, cv, and test
        self.data = [None] * 3
        # data[0] = training, data[1] = cv, data[2] = test
        self.data[0] = self.data_dict["training_data"]
        self.data[1] = self.data_dict["cv_data"]
        self.data[2] = self.data_dict["test_data"]

        # Extract labels and indicators
        self.indicators = [None] * 3
        self.labels = [None] * 3
        for i, dataset in enumerate(self.data):
            self.indicators[i] = np.vstack([x[0] for x in dataset])
            self.labels[i] = np.vstack([x[1] for x in dataset])

        training_predictions = np.vstack([[1, 0, 0] for x in self.labels[0]])
        training_error = Tinker.error(training_predictions, self.labels[0])
        cv_predictions = np.vstack([[1, 0, 0] for x in self.labels[1]])
        cv_error = Tinker.error(cv_predictions, self.labels[1])
        test_predictions = np.vstack([[1, 0, 0] for x in self.labels[2]])
        test_error = Tinker.error(test_predictions, self.labels[2])

        return [training_error, cv_error, test_error]

    def with_time_series(self, num_days, hidden_layer_size, reg):
        """Trains/tests using only a time series of the past num_days

        Uses the given network design parameters

        Returns a list of [training_error, cv_error, test_error]
        """
        # Load training, cv, and test
        self.data = [None] * 3
        # data[0] = training, data[1] = cv, data[2] = test
        self.data[0] = self.data_dict["training_data"]
        self.data[1] = self.data_dict["cv_data"]
        self.data[2] = self.data_dict["test_data"]

        # Extract labels and indicators
        self.indicators = [None] * 3
        self.labels = [None] * 3
        for i, dataset in enumerate(self.data):
            self.indicators[i] = np.vstack([x[0] for x in dataset])
            self.labels[i] = np.vstack([x[1] for x in dataset])

        # Filter time series down to num_days
        for i, dataset in enumerate(self.indicators):
            self.indicators[i] = np.vstack(
                [x[:num_days] for x in self.indicators[i]])

        # Instantiate network and train
        input_layer_size = num_days
        num_outputs = 3
        activation_function = soldier.sigmoid
        activation_function_gradient = soldier.sigmoid_gradient
        output_activation_func = soldier.sigmoid
        output_activation_func_gradient = soldier.sigmoid_gradient

        nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size,
                                   num_outputs, reg, activation_function,
                                   activation_function_gradient,
                                   output_activation_func,
                                   output_activation_func_gradient)

        nn.train(self.indicators[0], self.labels[0])

        training_predictions = nn.predict(self.indicators[0])[0]
        training_error = Tinker.error(training_predictions, self.labels[0])
        cv_predictions = nn.predict(self.indicators[1])
        cv_error = Tinker.error(cv_predictions, self.labels[1])
        test_predictions = nn.predict(self.indicators[2])
        test_error = Tinker.error(test_predictions, self.labels[2])

        return [training_error, cv_error, test_error]
