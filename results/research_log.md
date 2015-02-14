2015/02/08
----------
+ Ran `experiment.py` with to test the pure time series approach with all possible combinations of:
    * Hidden Layer Size: [1, 3, 10, 30, 100, 300]
    * Regularization: [0, 1, 3, 10, 30, 100, 300, 1000]
    * Number of previous days to use in time series: [1, 3, 10, 30, 100]
+ Determined that, using a pure time-series based approach, cross-validation accuracies converge to 33.3333% for a number of parameter combinations.
    * This suggests that a pure time-series based approach will bear very little fruit, as this is approximately the error that is to be expected if the neural network were to return [1, 1, 1] (buy, sell, *and* hold) for all examples. Such a prediction system indicates that the neural network has "given up", unable to detect a pattern in the data it was given, simply returns the same answer for all examples. The accuracy being dramatically higher on the training set suggests that the model suffers from high variance as well, which probably means that a pure time series approach uses too many features.
2015/02/09
----------
+ Switching to error instead of accuracy as the measure of neural network success.
