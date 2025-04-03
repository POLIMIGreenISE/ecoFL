import numpy as np

# The partitions must be set accordingly to the Data quality we want, so it must be set as a parameter that changes
def load_partition(idx, x_train, y_train, x_val, y_val, indexes_train, indexes_val):
    """Load train and test data to simulate a partition."""

    return (
        np.copy(x_train[indexes_train[idx] : indexes_train[idx+1]]),
        np.copy(y_train[indexes_train[idx] : indexes_train[idx+1]]),
    ), (
        np.copy(x_val[indexes_val[idx] : indexes_val[idx+1]]),
        np.copy(y_val[indexes_val[idx] : indexes_val[idx+1]]),
    )