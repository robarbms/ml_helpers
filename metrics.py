import numpy as np

# Calculates errors across 2 lists
def errors(predictions, actual):
    return np.array(predictions) - np.array(actual)

# Calculates mean average error
def mae(predictions, actual):
    return sum(errors(predictions, actual).tolist()) / len(predictions)

# Calculates mean squared error
def mse(prediction, actual):
    return sum(np.square(errors(prediction, actual)).tolist()) / len(prediction)
