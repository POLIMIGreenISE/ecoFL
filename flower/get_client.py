from flower.FlowerClient import FlowerClient
from data_processing.load_partition import load_partition
from data_processing.data_quality import data_quality_poisoning, clean_dataset
from models.resnet_model import build_ResNet
from flwr.common import Context
import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tfk = tf.keras


# Function to initialize flower node client with its data partition (x_train_p refers to the all x_train set from which the partition is computed)
# Per simulation only data quality dimension (together with or without data volume) can be poisoned
def get_client_fn_simulation(
    x_train_p: np.ndarray,
    y_train_p: np.ndarray,
    x_val_p: np.ndarray,
    y_val_p: np.ndarray,
    indexes_train: list,
    indexes_val: list,
    node_options_lst: list,
    recommender: bool = False
):
    # It must be called to create an instance of a new FlowerClient
    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        #Consider always the order of the node in the list, not the node assigned during configuration phase
        node_id = node_options_lst[(int)(context.node_config["partition-id"])]['node_id']

        # Note: each client gets a different trainloader/valloader, so each client will train and evaluate on their own unique data
        print(
            "Client with Node ID: {}\n".format(node_id)
        )

        (x_train, y_train), (x_test, y_test) = load_partition(
            idx=node_id,
            x_train=x_train_p,
            y_train=y_train_p,
            x_val=x_val_p,
            y_val=y_val_p,
            indexes_train=indexes_train,
            indexes_val=indexes_val,
        )


         # Poisoning of the node training data
        x_train, y_train = data_quality_poisoning(
            x_train=x_train,
            y_train=y_train,
            options=next((node['data'] for node in node_options_lst if node['node_id'] == node_id), None) 
        )
        # Smart Reduction of the Dataset -> only for Recommender Mode
        if recommender:
            x_train = np.squeeze(x_train)
            x_train, y_train = clean_dataset(
                dataset_percent=next((node['data']['data_volume']['data_quality_dimension_percentage_sr'] for node in node_options_lst if node['node_id'] == node_id), None), 
                x_train=x_train, 
                y_train=y_train, 
                extend=True)
            x_train = np.expand_dims(x_train, axis=-1)
        y_train = tfk.utils.to_categorical(y_train) 

        # Load model
        model_client = build_ResNet(x_train.shape[1:], y_train.shape[-1])

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            cid=node_id,
            model=model_client,
            batch_size=next((node['training']['batch_size'] for node in node_options_lst if node['node_id'] == node_id), 64),
            local_epochs=next((node['training']['epochs'] for node in node_options_lst if node['node_id'] == node_id), 64),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dev_energy_consumption=next((node.get('energy', {}).get('energy_consumption') for node in node_options_lst if node['node_id'] == node_id), None),
            dev_emission=next((node.get('energy', {}).get('carbon_intensity') for node in node_options_lst if node['node_id'] == node_id), None),
        ).to_client()

    return client_fn
