import flwr as fl
from flwr.client import NumPyClient

import tensorflow as tf
tfk = tf.keras
import time
import json
import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from typing import Dict, List, Optional, Tuple, Union

from data_processing.data_quality import data_quality_poisoning

#Flower Client which defines the Nodes Clients
class FlowerClient(NumPyClient):
    def __init__(self, cid, model, batch_size, local_epochs, x_train, y_train, x_test, y_test, dev_energy_consumption = None, dev_emission = None):
        self.cid = cid
        self.model = model
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.dev_energy_consumption, self.dev_emission = dev_energy_consumption, dev_emission
        self.measure_parameters  = {
              'effective_epochs': 0,
              'effective_emissions_kg': 0,
              'effective_energy_consumed': 0,
              'effective_duration': 0
        }

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        # Get parameters as a list of NumPy ndarray's
        return self.model.get_weights()


    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("Starting Fit")

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size = int(min(self.x_train.shape[0] / 10, self.batch_size))   

        start_time = time.time()
        with OfflineEmissionsTracker(country_iso_code="ITA", project_name=1, log_level='warning', tracking_mode='process', allow_multiple_runs=True) as tracker:
            # Train the model using hyperparameters from config
            history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=batch_size,
                epochs=self.local_epochs,
                validation_split=.1,
                callbacks=[
                    tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True),
                    tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=20, factor=0.5,
                                                    min_lr=0.0001),
                ]
            )

        #Read emission.csv to extract emissions data
        emissions_df = pd.read_csv('emissions.csv')
        self.measure_parameters['effective_duration'] = time.time() - start_time
        self.measure_parameters['effective_epochs'] = len(history.epoch)
        
        if self.dev_energy_consumption and self.dev_energy_consumption:
            self.measure_parameters['effective_energy_consumed'] = round(((self.measure_parameters['effective_duration']/3600)*(self.dev_energy_consumption / 100)), 3)
            self.measure_parameters['effective_emissions_kg'] = round(((self.measure_parameters['effective_energy_consumed']*self.dev_emission) / 100), 3)
        else:
            #Why is it necessary the division per 100?
            self.measure_parameters['effective_energy_consumed'] = emissions_df['energy_consumed'].iloc[-1]
            self.measure_parameters['effective_emissions_kg'] = emissions_df['emissions'].iloc[-1]
        print(self.measure_parameters)
        
        
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
            "emission_kg": self.measure_parameters['effective_emissions_kg'],
            "energy_consumed": self.measure_parameters['effective_energy_consumed'],
            "duration": self.measure_parameters['effective_duration'],
            "epochs": self.measure_parameters['effective_epochs'],
        }

        # Build and return response
        return self.model.get_weights(), num_examples_train, results

    def evaluate(self, parameters, config):
        print(f"Client {self.cid} evaluation")
        """Evaluate parameters on the locally held test set."""

        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)

        num_examples_test = len(self.x_test)
        # Build and return response
        return float(loss), num_examples_test, {"accuracy": float(accuracy)}