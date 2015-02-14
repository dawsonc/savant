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
        """Returns the % error of the network on the given examples"""
        pct_error = 1 - np.mean(np.double(np.equal(predictions, labels)))
        pct_error = pct_error * 100

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
        training_error = Tinker.error(
            training_predictions, self.labels[0])
        cv_predictions = np.vstack([[1, 0, 0] for x in self.labels[1]])
        cv_error = Tinker.error(cv_predictions, self.labels[1])
        test_predictions = np.vstack([[1, 0, 0] for x in self.labels[2]])
        test_error = Tinker.error(test_predictions, self.labels[2])

        return [training_error, cv_error, test_error]

    def with_time_series(self, num_days, hidden_layer_size, reg,
                         num_examples=-1):
        """Trains/tests using only a time series of the past num_days

        Uses the given network design parameters
        Trains on the first m examples of the training set, to facilitate
            learning curve analysis (defaults to using all the training set)

        Returns a list of [training_error, cv_error, test_error]
        """
        # Load training, cv, and test
        self.data = [None] * 3
        # data[0] = training, data[1] = cv, data[2] = test
        self.data[0] = self.data_dict["training_data"][:num_examples]
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
        training_error = Tinker.error(
            training_predictions, self.labels[0])
        cv_predictions = nn.predict(self.indicators[1])
        cv_error = Tinker.error(cv_predictions, self.labels[1])
        test_predictions = nn.predict(self.indicators[2])
        test_error = Tinker.error(test_predictions, self.labels[2])

        return [training_error, cv_error, test_error]

    def with_indicators(self, hidden_layer_size, reg, num_examples=-1):
        """Trains/tests using only a collection of technical indicators

        Uses the given network design parameters
        Trains on the given number of training examples

        Returns (errors, normalizations), where
            errors = a list of [training_error, cv_error, test_error]
            normalizations = a list of tuples of (mean, std) for each indicator
        """
        # Load training, cv, and test
        self.data = [None] * 3
        # data[0] = training, data[1] = cv, data[2] = test
        self.data[0] = self.data_dict["training_data"][:num_examples]
        self.data[1] = self.data_dict["cv_data"]
        self.data[2] = self.data_dict["test_data"]

        # Extract labels and indicators
        self.indicators = [None] * 3
        self.labels = [None] * 3
        for i, dataset in enumerate(self.data):
            self.indicators[i] = np.vstack([x[0] for x in dataset])
            self.labels[i] = np.vstack([x[1] for x in dataset])

        # Map time series indicators to technical indicators
        def calculate_indicators(ts):
            def stochastic_k(n):
                return 100 * (ts[n] - np.min(ts[n:n + 14])) / \
                    (np.max(ts[n:n + 10]) - np.min(ts[n:n + 10]))

            def stochastic_d(n):
                return (stochastic_k(n) +
                        stochastic_k(n + 1) +
                        stochastic_k(n + 2)) / 3

            def momentum(n):
                return ts[n] - ts[n + 1]

            def price_rate_of_change(n):
                return ts[n] / ts[n + 1] * 100

            def williams_pct_r(n):
                return -100 * (np.max(ts[n:n + 5]) - ts[n]) / \
                              (np.max(ts[n:n + 5]) - np.min(ts[n:n + 5]))

            def disparity(n, moving_avg_length):
                return 100 * ts[n] / np.mean(ts[n:n + moving_avg_length])

            def price_oscillator(n):
                return (np.mean(ts[n:n + 5]) - np.mean(ts[n:n + 10])) / \
                    np.mean(ts[n:n + 5])

            def commodity_channel_index(n):
                # This is a bastardized version of the CCI, as we don't have
                # . intraday Highs & Lows (yet)
                D = np.sum(np.abs(ts[n:n + 20] - np.mean(ts[n:n + 20]))) / 20
                return (ts[n] - np.mean(ts[n:n + 20])) / (0.015 * D)

            def relative_strength_index():
                # This is the most complicated indicator to calculate
                # . so watch carefully

                # Calculate interday changes
                deltas = ts[:-1] - ts[1:]

                # Calculate ups and downs
                ups = np.hstack([x if x > 0 else 0 for x in deltas])
                downs = np.hstack([abs(x) if x < 0 else 0 for x in deltas])

                # Find the 10-day exponential moving average
                # . for both the ups and the downs
                alpha = 2.0 / (1.0 + 20.0)
                ups_ema = 1.0
                for price in reversed(ups):
                    ups_ema = (price * alpha) + (ups_ema * (1 - alpha))
                downs_ema = 1.0
                for price in reversed(downs):
                    downs_ema = (price * alpha) + (downs_ema * (1 - alpha))

                # Compute relative strength factor
                rs = ups_ema / downs_ema

                # Return relative strength index
                return 100 - 100 / (1 + rs)

            def macd(n):
                alpha = 0.1
                ema12 = 1
                for price in reversed(ts[n:n + 12 + 1]):
                    ema12 = alpha * (price - ema12) + ema12
                ema26 = 1
                for price in reversed(ts[n:n + 26 + 1]):
                    ema26 = alpha * (price - ema26) + ema26

                return ema12 - ema26

            def macd_hist():
                alpha = 0.2
                signal_line = 1  # 9-day EMA of MACD
                for i in range(9):
                    signal_line = alpha * (macd(i) - signal_line) + signal_line

                return macd(0) - signal_line

            def adx():
                """Calculate Average Directional Index"""
                # Calculate interday changes
                deltas = ts[:-1] - ts[1:]

                # Calculate ups and downs
                ups = np.hstack([x if x > 0 else 0 for x in deltas])
                downs = np.hstack([abs(x) if x < 0 else 0 for x in deltas])

            # Slow stochastic % D is the 3-day moving average of stochastic D
            slow_stochastic_d = (stochastic_d(0) + stochastic_d(1) +
                                 stochastic_d(2)) / 3

            # 5-day moving average
            five_day_ma = np.mean(ts[0:6])

            most_recent_price = ts[0]

            return np.array([most_recent_price,
                             five_day_ma,
                             # stochastic_k(0),
                             stochastic_d(0),
                             slow_stochastic_d,
                             macd_hist(),
                             # momentum(0),
                             # price_rate_of_change(0),
                             # williams_pct_r(0),
                             # disparity(0, 5),
                             # disparity(0, 10),
                             # price_oscillator(0),
                             # commodity_channel_index(0),
                             relative_strength_index()
                             ])

        for i, dataset in enumerate(self.indicators):
            self.indicators[i] = np.vstack(
                [calculate_indicators(x) for x in dataset])

        # Normalize indicators based on training indicators
        training_means = self.indicators[0].mean(0)
        training_range = self.indicators[0].ptp(0)

        for i, dataset in enumerate(self.indicators):
            self.indicators[i] = self.indicators[i] - training_means
            self.indicators[i] = self.indicators[i] / training_range

        # Instantiate network and train
        input_layer_size = self.indicators[0].shape[1]
        num_outputs = 1
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
        training_error = Tinker.error(
            training_predictions, self.labels[0])
        cv_predictions = nn.predict(self.indicators[1])
        cv_error = Tinker.error(cv_predictions, self.labels[1])
        test_predictions = nn.predict(self.indicators[2])
        test_error = Tinker.error(test_predictions, self.labels[2])

        return [training_error, cv_error, test_error], \
            (training_means, training_range)
