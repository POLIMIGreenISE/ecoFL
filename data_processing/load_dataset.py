import numpy as np
import json
from sklearn.utils import shuffle

def load_dataset(dataset_train_path: str,
                 dataset_test_path: str):
    
    train_data = np.loadtxt(dataset_train_path)
    test_data = np.loadtxt(dataset_test_path)

    #np.random.shuffle(train_data)
    #Swap x_train and x_test because the test set is bigger
    X_train = np.expand_dims(train_data[:, 1:],axis=-1)
    y_train = train_data[:, 0]

    X_test = np.expand_dims(test_data[:, 1:], axis=-1)
    y_test = test_data[:, 0]

    #Take the 10% as validation set
    validation_sample = (int)(0.1 * X_train.shape[0])

    #Take the last 1000 samples of x_train for the validation step
    X_val = X_train[-validation_sample:]
    y_val = y_train[-validation_sample:]

    X_train = X_train[:-validation_sample]
    y_train = y_train[:-validation_sample]

    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def splitting_dataset(dataset: np.ndarray, num_clients: int, data_volume_lst: list = [], recommender: bool = False):
  total_points = dataset.shape[0]
  indexes = [0]

  if recommender:
    #Retrieve splitting from configuration according to the data_volume parameter defined
    for data_volume in data_volume_lst:
      number_points = (int)(data_volume*total_points)
      new_index = indexes[-1] + number_points
      indexes.append(new_index)
    
  else:
    #Uniform splitting
    single_set = int(total_points/num_clients)
    for client in range(0,num_clients):
      new_index = indexes[-1] + single_set
      indexes.append(new_index)

  return indexes