# Data quality libraries
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from data_processing.load_partition import load_partition


## DROPPING DATA: DROPS DATA
def reduce_data_volume(x_train, y_train, options):
    completeness_percentage = options["data_quality_dimension_percentage"]
    method = options["experiment_method"]

    dropped_percentage = 1 - completeness_percentage

    if method == "uniform":
        drop_mask = np.random.choice(
            [True, False],
            x_train.shape[0],
            p=[1 - dropped_percentage, dropped_percentage],
        )
        return x_train[drop_mask], y_train[drop_mask]

    n_original = x_train.shape[0]
    total_to_drop = int(n_original * dropped_percentage)
    classes, classes_counts = np.unique(y_train, return_counts=True)
    classes_counts_original = dict(zip(classes, classes_counts))

    classes_counts = classes_counts_original.copy()

    sampling_strategy = {i: 0 for i in classes}
    class_to_drop = max(classes_counts_original, key=classes_counts_original.get)

    sampling_strategy[class_to_drop] += total_to_drop
    classes_counts[class_to_drop] -= total_to_drop

    resulting_strategy = {}
    for class_ in classes_counts_original.keys():
        resulting_strategy[class_] = (
            classes_counts_original[class_] - sampling_strategy[class_]
        )

    sampler = RandomUnderSampler(sampling_strategy=resulting_strategy)

    rebalanced_x_train, rebalanced_y_train = sampler.fit_resample(
        np.squeeze(x_train), y_train
    )

    permutations = np.random.permutation(rebalanced_x_train.shape[0])
    return (
        np.expand_dims(rebalanced_x_train[permutations], -1),
        rebalanced_y_train[permutations],
    )


## CONSISTENCY: DUPLICATES AND CHANGES LABELS
def reduce_consistency(x_train, y_train, options):
    consistency_percentage = options["data_quality_dimension_percentage"]
    method = options["experiment_method"]
    class_list = set(np.unique(y_train))
    original_total = x_train.shape[0]
    n_to_add = int(original_total - consistency_percentage * original_total)
    if method == "uniform":
        indices = np.arange(0, original_total)
    else:
        classes, classes_counts = np.unique(y_train, return_counts=True)
        classes_counts_original = dict(zip(classes, classes_counts))
        class_to_corrupt = max(classes_counts_original, key=classes_counts_original.get)
        indices = np.squeeze(np.argwhere(y_train == class_to_corrupt))

    # Select indices to remove (original data points)
    indices_to_remove = np.random.choice(indices, n_to_add)
    
    # Create mask to keep only the data points we want to retain
    mask = np.ones(original_total, dtype=bool)
    mask[indices_to_remove] = False
    
    # Get the data points we want to keep
    x_train_retained = x_train[mask]
    y_train_retained = y_train[mask]

    # Create inconsistent duplicates
    inconsistent_indices = np.random.choice(indices, n_to_add)
    x_train_duplicates, y_train_duplicates = (
        x_train[inconsistent_indices].copy(),
        y_train[inconsistent_indices].copy(),
    )
    for i in range(x_train_duplicates.shape[0]):
        y_value = y_train_duplicates[i]
        new_y_value = np.random.choice(list(class_list.difference({y_value})))
        y_train_duplicates[i] = new_y_value

    # Concatenate retained data with inconsistent duplicates
    new_x_train, new_y_train = np.concatenate(
        (x_train_retained, x_train_duplicates)
    ), np.concatenate((y_train_retained, y_train_duplicates))
    permutations = np.random.permutation(new_x_train.shape[0])

    return new_x_train[permutations], new_y_train[permutations]


## ACCURACY: CHANGES LABELS
def reduce_accuracy(x_train, y_train, options):

    accuracy_percentage = options["data_quality_dimension_percentage"]
    method = options["experiment_method"]
    corrupted_percentage = 1 - accuracy_percentage
    total_amount = x_train.shape[0]
    class_list = set(np.unique(y_train))

    if method == "uniform":
        for i in range(total_amount):
            if np.random.random() < corrupted_percentage:
                y_value = y_train[i]
                new_y_value = np.random.choice(list(class_list.difference({y_value})))
                y_train[i] = new_y_value

        return x_train, y_train

    classes, classes_counts = np.unique(y_train, return_counts=True)
    classes_counts_original = dict(zip(classes, classes_counts))
    class_to_corrupt = max(classes_counts_original, key=classes_counts_original.get)
    class_percentage = classes_counts_original[class_to_corrupt] / total_amount
    other_classes = list(class_list.difference({class_to_corrupt}))

    class_percentage_to_corrupt = corrupted_percentage / class_percentage
    for i in range(total_amount):
        y_value = y_train[i]
        if (
            y_value == class_to_corrupt
            and np.random.random() < class_percentage_to_corrupt
        ):

            new_y_value = np.random.choice(other_classes)
            y_train[i] = new_y_value

    return x_train, y_train


## COMPLETENESS: Deletes inner points
def reduce_completeness(x_train, y_train, options):
    accuracy_percentage = options["data_quality_dimension_percentage"]
    method = options["experiment_method"]
    inner_missing_percentage = 0.4

    number_of_sequences, length_of_sequence, dimensions = x_train.shape
    number_of_missing_points_per_sequence = int(
        inner_missing_percentage * length_of_sequence
    )

    total_amount = x_train.shape[0]

    if method == "uniform":
        while True:
            affected_percentage = (1 - accuracy_percentage) / inner_missing_percentage
            if affected_percentage <= 1:
                break

            inner_missing_percentage *= 1.02
        for i in range(total_amount):
            if np.random.random() < affected_percentage:
                start = np.random.randint(
                    0, length_of_sequence - number_of_missing_points_per_sequence
                )
                x_train[i][start : start + number_of_missing_points_per_sequence] = (
                    np.mean(x_train[i])
                )

        return x_train, y_train

    class_list = set(np.unique(y_train))
    classes, classes_counts = np.unique(y_train, return_counts=True)
    classes_counts_original = dict(zip(classes, classes_counts))
    class_to_corrupt = max(classes_counts_original, key=classes_counts_original.get)
    class_percentage = classes_counts_original[class_to_corrupt] / total_amount
    affected_class_percentage = min(
        1, (1 - accuracy_percentage) / (inner_missing_percentage * class_percentage)
    )
    while True:
        actual_accuracy_percentage = (
            1 - inner_missing_percentage * class_percentage * affected_class_percentage
        )

        if actual_accuracy_percentage <= accuracy_percentage:
            break

        inner_missing_percentage *= 1.02
        affected_class_percentage = min(
            1, (1 - accuracy_percentage) / (inner_missing_percentage * class_percentage)
        )

    number_of_missing_points_per_sequence = int(
        inner_missing_percentage * length_of_sequence
    )
    for i in range(total_amount):
        y_value = y_train[i]
        if (
            y_value == class_to_corrupt
            and np.random.random() < affected_class_percentage
        ):
            start = np.random.randint(
                0, length_of_sequence - number_of_missing_points_per_sequence
            )
            x_train[i][start : start + number_of_missing_points_per_sequence] = np.mean(
                x_train[i]
            )

    return x_train, y_train


def reduce_completeness_nan(x_train, y_train, options):
    accuracy_percentage = options["data_quality_dimension_percentage"]
    inner_missing_percentage = 0.4

    number_of_sequences, length_of_sequence, dimensions = x_train.shape
    number_of_missing_points_per_sequence = int(
        inner_missing_percentage * length_of_sequence
    )

    total_amount = x_train.shape[0]

    while True:
        affected_percentage = (1 - accuracy_percentage) / inner_missing_percentage
        if affected_percentage <= 1:
            break

        inner_missing_percentage *= 1.02
    for i in range(total_amount):
        if np.random.random() < affected_percentage:
            start = np.random.randint(
                0, length_of_sequence - number_of_missing_points_per_sequence
            )
            # Instead of filling with the mean to do data imputation, insert NaN
            x_train[i][start : start + number_of_missing_points_per_sequence] = np.nan

    return x_train, y_train


# Compute the consistency of the dataset
def dataset_consistency(x_train):
    amount_of_duplicates = np.sum(np.unique(x_train, axis=0, return_counts=True)[1] - 1)

    # Assuming the duplicates have different classes
    return 1 - amount_of_duplicates / x_train.shape[0]


# Compute the completeness of the dataset
def dataset_completeness(x_train):
    return 1 - np.count_nonzero(np.isnan(x_train)) / np.prod(x_train.shape)


def clean_dataset(dataset_percent, x_train, y_train, extend):

    goal_size = int(x_train.shape[0] * dataset_percent)

    # In order of most important DQ dimensions

    # Consistency
    current_consistency = dataset_consistency(x_train)
    if current_consistency < 1:
        left_to_remove = x_train.shape[0] - goal_size

        repeats = filter(
            lambda x: x.shape[0] > 1,
            pd.DataFrame(x_train).groupby([0]).indices.values(),
        )
        current_number_take_out = 0
        take_out_indices = []

        for repeat_index in repeats:
            current_number_take_out += repeat_index.shape[0]
            take_out_indices = [*take_out_indices, *repeat_index]

            if current_number_take_out > left_to_remove:
                break

        x_train = np.delete(x_train, take_out_indices, axis=0)
        y_train = np.delete(y_train, take_out_indices, axis=0)

    # Completeness
    left_to_remove = x_train.shape[0] - goal_size

    nan_mask = np.squeeze(np.any(np.isnan(x_train), axis=1))
    number_to_drop_from_nan_mask = np.sum(nan_mask)

    """
  if number_to_drop_from_nan_mask > left_to_remove:
    # If the number of values with nan is greater than how many we need to remove based on goal
    # Turn some of the nan mask to false, to keep some nans
    change_to_false_indices = random.sample(np.where(nan_mask == True)[0].tolist(),number_to_drop_from_nan_mask - left_to_remove)
    nan_mask[change_to_false_indices] = False
  """

    x_train = x_train[~nan_mask]
    y_train = y_train[~nan_mask]

    # It removes randomly the data after it has reached the best values of consistencies and completeness
    # Remove till get goal
    # Remove this part because we have no goal_size costraint

    if extend:
        if x_train.shape[0] > goal_size:
            # randomly remove data
            dropped_percentage = (x_train.shape[0] - goal_size) / x_train.shape[0]
            drop_mask = np.random.choice(
                [True, False],
                x_train.shape[0],
                p=[1 - dropped_percentage, dropped_percentage],
            )

            # Border Case - Ignore
            if not np.any(drop_mask):
                drop_mask[0] = True

            x_train = x_train[drop_mask]
            y_train = y_train[drop_mask]
    """
  #full_dataset_train = np.concatenate((np.expand_dims(self.y_train, axis=1), np.squeeze(self.x_train)), axis=1, dtype='str')
  #text_dataset = "\n".join([" ".join(i) for i in full_dataset_train])
  """

    return x_train, y_train


def compute_effective_percentage(
    node, x_train, y_train, x_val, y_val, indexes_train, indexes_val
):
    (x_train, y_train), (_, _) = load_partition(
        node['node_id'], x_train, y_train, x_val, y_val, indexes_train, indexes_val
    )
    initial_dataset_length = len(x_train)
    # Retrieve the dataset in the original form
    options_consistency = {
        "data_quality_dimension_percentage": node["data"]["data_consistency"]['data_quality_dimension_percentage'],
        "experiment_method": "uniform",
    }
    options_completeness = {
        "data_quality_dimension_percentage": node["data"]["data_completeness"]['data_quality_dimension_percentage'],
        "experiment_method": "uniform",
    }
    x_train_d, y_train_d = reduce_completeness_nan(
        x_train, y_train, options_completeness
    )
    x_train_d, y_train_d = reduce_consistency(x_train, y_train, options_consistency)

    # Clean dataset with only smart reduction
    x_train_d = np.squeeze(x_train_d)
    x_train_c, y_train_c = clean_dataset(0.0, x_train_d, y_train_d, False)
    x_train_c = np.expand_dims(x_train_c, axis=-1)

    final_dataset_length = len(x_train_c)
    final_percentage_node = round(
        ((node["data"]["data_volume"]["data_quality_dimension_percentage_overall"] * final_dataset_length) / initial_dataset_length), 2
    )

    return final_percentage_node


def data_quality_poisoning(
    x_train: np.ndarray,
    y_train: np.ndarray,
    options: dict,
):
    # Reduce Data Volume
    if options["data_volume"]["data_quality_dimension_percentage"] < 1.00:
        x_train, y_train = reduce_data_volume(x_train, y_train, options["data_volume"])
        print(
            f"Data Volume reduced by {float(format((1.00 - options['data_volume']['data_quality_dimension_percentage']), '.2f'))*100} % "
        )
    # Reduce Data Consistency
    if options["data_consistency"]["data_quality_dimension_percentage"] < 1.00:
        x_train, y_train = reduce_consistency(x_train, y_train, options["data_consistency"])
        print(
            f"Data Consistency reduced by {float(format((1.00 - options['data_consistency']['data_quality_dimension_percentage']), '.2f'))*100} % "
        )
    # Reduce Data Accuracy
    if options["data_accuracy"]["data_quality_dimension_percentage"] < 1.00:
        x_train, y_train = reduce_accuracy(x_train, y_train, options["data_accuracy"])
        print(
            f"Data Accuracy reduced by {float(format((1.00 - options['data_accuracy']['data_quality_dimension_percentage']), '.2f'))*100} % "
        )
    # Reduce Data Accuracy
    if options["data_completeness"]["data_quality_dimension_percentage"] < 1.00:
        x_train, y_train = reduce_completeness_nan(
            x_train, y_train, options["data_completeness"]
        )
        print(
            f"Data Completeness reduced by {float(format((1.00 - options['data_completeness']['data_quality_dimension_percentage']), '.2f'))*100} % "
        )

    return x_train, y_train
