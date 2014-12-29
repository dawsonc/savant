"""Module for reshaping the historical data"""

def reduce_to_column(data, column_key):
    """Reduces the csv data by removing all columns except the given column

    pandas.DataFrame, String -> pandas.Series
    """
    try:
        return data[column_key]
    except KeyError:
        raise ValueError("Data must contain the column %s" % column_key)

def get_feature_vector(data, index, k=30):
    """Given historical data, constructs a feature vector for the day indicated by index

    Arguments:
        data  --- pandas.Series containing historical price data for a particular security
        index --- integer indicating which row for which to construct a feature vector
                    Note: 0 is the most recent row, higher indices are further in the past
        k     --- integer indicating the number of preceeding rows to include in the feature vector

    Returns the tuple (features, result) such that:
        features is a numpy array containing the prices for days (index + 30) through (index + 1)
        result is an double containing the price for day (index)
    """
    result = data[index]

    starting_index = index + 1
    end_index = index + k
    features = data[starting_index:end_index+1]

    return (features, result)
