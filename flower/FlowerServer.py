#import flwr libraries
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    EvaluateRes,
    EvaluateIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING
import random

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
plt.rc('font', size=16)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tfk = tf.keras
tfkl = tf.keras.layers


class FedCustom(Strategy):
    def __init__(
        self,
        file_path,
        model,
        initial_parameters,
        x_test,
        y_test,
        n_rounds: str,
        node_selection_strategy_type: str,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.initial_parameters = initial_parameters
        self.x_test = x_test
        self.y_test = y_test
        self.n_rounds = n_rounds
        self.node_selection_strategy_type = node_selection_strategy_type
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.measure_parameters  = {
              'effective_epochs': 0,
              'effective_emissions_kg': 0,
              'effective_energy_consumed': 0,
              'effective_duration': 0
        }

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        #This method is called at the beginning of each round in order to choose and select the clients to be trained
        # Sample clients

        if self.node_selection_strategy_type == 'basic':
            '''

            This strategy at the beginning of the first round chooses a sample of clients to run a federated learning training. 
            Each round a new set of clients of the same sample size will be taken for training
            There is no data poisoning at this step.

            '''
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            print(f"Round {server_round} will sample clients: {[client.cid for client in clients]}")

            standard_config = {}
            fit_configurations = []
            for idx, client in enumerate(clients):
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            return fit_configurations
        

        if self.node_selection_strategy_type == 'vertical':
            '''

            This strategy at the beginning of the first round chooses a sample (defined by sample_size equal to fraction defined) of clients to run a federated learning training. 
            The selected set will remain the same for all rounds.
            Only a nodes subset of the set selected is poisoned by a given data quality dimensions, the other set is kept cleaned

            '''
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            if server_round == 1:
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
                print(f"Round {server_round} will sample clients: {[client.node_id for client in clients]}")
            # Otherwise I select the same clients as the previous round
            else:
                clients = [cli for cli in self.prev_clients]
                print(f"Round {server_round} will resample clients: {[client.node_id for client in clients]}")

            self.prev_clients = clients

            fit_configurations = []
            config_affected = {"batch_size": 64, "local_epochs": 40, "quality_percentage": 0.5, "round": server_round}
            config_not_affected = {"batch_size": 64, "local_epochs": 40, "quality_percentage": 1.0, "round": server_round}
            n_clients = len(clients)
            num_clients_affected = n_clients - (int(n_clients*self.vertical_quality))
            #Take the first num_clients_affected from the list of clients selected randomly at the beginning
            clients_affected = clients[:num_clients_affected]
            for client in clients_affected:
                fit_configurations.append((client, FitIns(parameters, config_affected)))

            clients_not_affected = clients[num_clients_affected:]
            for client in clients_not_affected:
                fit_configurations.append((client, FitIns(parameters, config_not_affected)))

            return fit_configurations
        

        if self.node_selection_strategy_type == 'set':
            '''

            This strategy at the beginning of the first round chooses a sample of clients to run a federated learning training. 
            The selected set will remain the same for all rounds but after each round only a subset from the chosen set is used for training
            There is no data poisoning at this step.

            '''
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            
            #TODO: Set as parameter
            size_all = 8

            if server_round == 1:
                #Define the fixed set of clients to be trained
                self.sample_clients = random.sample(range(0, 8), self.num_samples)

                #Retrieve all the clients
                clients_all = client_manager.sample(
                    num_clients=size_all, min_num_clients=min_num_clients
                )

                #Retrieve the clients that corresponds to set defined before
                for cli in clients_all:
                    if cli.cid in self.sample_clients:
                        self.set_clients
                self.set_clients = [cli for cli in clients_all if int(cli.cid) in self.sample_clients]
                print(f"The initial set of clients is {self.sample_clients}")

            clients = random.sample(self.set_clients, sample_size)
            print(f"Round {server_round} will sample clients: {[client.cid for client in clients]}")


            # Create custom configs in order to change and personalize the config per each client
            standard_config = {"batch_size": 64, "local_epochs": 40}
            fit_configurations = []
            for idx, client in enumerate(clients):
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            print("Config ended")
            return fit_configurations

        
        if self.node_selection_strategy_type == 'fixed':
            '''

            This strategy is equal to Vertical strategy but with no data poisoning.

            '''
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            if server_round == 1:
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
                print(f"Round {server_round} will sample clients: {[client.cid for client in clients]}")
            # Otherwise I select the same clients as the previous round
            else:
                clients = [cli for cli in self.prev_clients]
                print(f"Round {server_round} will resample clients: {[client.cid for client in clients]}")

            # record client proxies
            self.prev_clients = clients

            # Create custom configs in order to change and personalize the config per each client
            standard_config = {}
            fit_configurations = []
            for idx, client in enumerate(clients):
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            return fit_configurations


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        #Compute emission and duration
        for _, res in results:
          self.measure_parameters['effective_emissions_kg'] += res.metrics['emission_kg']
          self.measure_parameters['effective_energy_consumed'] += res.metrics['energy_consumed']
          self.measure_parameters['effective_duration'] += res.metrics['duration']
          self.measure_parameters['effective_epochs'] += res.metrics['epochs']

        #If the server round is the last, save results
        if(server_round == self.n_rounds):
          with open(self.file_path, "r") as json_file:
            data = json.load(json_file)

          last_obj = data[-1]
          last_obj['effective_emissions_kg'] = self.measure_parameters['effective_emissions_kg']
          last_obj['effective_energy_consumed'] = self.measure_parameters['effective_energy_consumed']
          last_obj['effective_duration'] = self.measure_parameters['effective_duration']
          last_obj['effective_epochs'] = self.measure_parameters['effective_epochs']
          print(f"Final records: {self.measure_parameters}")

          with open(self.file_path, "w") as json_file:
            json.dump(data, json_file)

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated
    

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {"server_round": server_round,
                  "file_path": self.file_path}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        self.model.set_weights(parameters_ndarrays)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

        with open(self.file_path, "r") as json_file:
            data = json.load(json_file)

        last_obj = data[-1]
        last_obj[f"Accuracy_{server_round}"] = accuracy

        with open(self.file_path, "w") as json_file:
            json.dump(data, json_file)

        return loss, {"accuracy": accuracy}
           

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients