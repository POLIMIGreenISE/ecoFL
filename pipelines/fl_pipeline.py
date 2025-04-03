import os
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from flwr.common import (
    ndarrays_to_parameters,
)
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents    
from flwr.simulation import run_simulation
from flwr.common import Context

from models.resnet_model import build_ResNet
from flower.FlowerServer import FedCustom
from flower.get_client import get_client_fn_simulation
from utils.utils import (
    build_results_schema,
    write_instance_on_results,
    read_main_experiment_results
)


def run_fl_simulation(
    data: dict,
    node_options_lst: list[dict],
    node_selection: str,
    fraction_fit: float,
    number_clients: int,
    rounds: int,
    file_path: str,
    recommender: bool = False
):
    """
    Runs the Federated Learning Simulation pipeline
    
    Args:
        data: Dictionary containing dataset splits
        node_options_lst: List of dictionaries containing information about each node clients
        node_selection: Strategy for node selection during Federated Learning training. It can be between the following two values: 'basic','fixed'.
        fraction_fit: Fraction of clients to use for training
        num_clients: Number of clients belonging to the federation
        rounds: Number of training rounds
        file_path: Path to save results

    Return:
        results_dct: Dictionary with results of the simulation
    """

    # TODO: move this step!
    y_val = tf.keras.utils.to_categorical(data["y_val"])
    y_test = tf.keras.utils.to_categorical(data["y_test"])

    input_shape = data["X_train"].shape[1:]
    classes = tf.keras.utils.to_categorical(data["y_train"]).shape[-1]

    result_schema = build_results_schema(
        experiment_type=node_selection,
        num_clients_available=int(number_clients),
        num_clients_trained=int(number_clients * fraction_fit),
        node_options_lst=node_options_lst
    )

    write_instance_on_results(file_path=file_path, result_schema=result_schema)

    # Build model
    new_model = build_ResNet(input_shape, classes)

    # Configure FL strategy
    strategy_basic = FedCustom(
        file_path=file_path,
        model=new_model,
        x_test=data["X_test"],
        y_test=y_test,
        n_rounds=rounds,
        node_selection_strategy_type=node_selection,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=number_clients,
        initial_parameters=ndarrays_to_parameters(new_model.get_weights()),
    )

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(
            config=config,
            strategy=strategy_basic,
        )

    # Create server and client apps
    server = ServerApp(server_fn=server_fn)
    client = ClientApp(
        client_fn=get_client_fn_simulation(
            x_train_p=data["X_train"],
            y_train_p=data["y_train"],
            x_val_p=data["X_val"],
            y_val_p=y_val,
            indexes_train=data["indexes_train"],
            indexes_val=data["indexes_val"],
            node_options_lst=node_options_lst,
            recommender=recommender
        )
    )

    # Run simulation
    backend_config = {"client_resources": {"num_cpus": os.cpu_count()}}
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=number_clients,
        backend_config=backend_config,
    )

    results_dct = read_main_experiment_results(file_path=file_path)
    
    return results_dct

