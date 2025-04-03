import random
import numpy as np
from joblib import load
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pipelines.fl_pipeline import run_fl_simulation
from models.resnet_model import build_ResNet
from methods.recommender_methods import (
    nodes_selection,
    nodes_selection_smart,
    nodes_selection_smart_minimal
)
import tensorflow as tf
from config import CONFIG

def run_fl_recommender(
        data: dict,
        node_options_lst: list[dict],
        recommender_method: str,
        accuracy_goal: float,
        rounds: int,
        dir_path: str
):
    """
    Runs the Federated Learning Recommender pipeline
    
    Args:
        data: Dictionary containing dataset splits
        node_options: List of dictionaries containing information about each node clients
        recommender_method: Method to be used for node selection and data reduction. The possible values are: 'NS', 'SR' and 'MSR' (they correspond to 'Node Selection', 'Smart Reduction', 'Minimal Smart Reduction')
        accuracy_goal: Accuracy 
        rounds: Number of training rounds
        dir_path: Dir Path to save results

    Return:
        results_dct: Dictionary with results of the simulation
    """
    logging.info(f"The Clients Info are: {node_options_lst}")
    #Run One Node Simulation
    single_node_lst = [random.choice(node_options_lst)]
    logging.info(f"The Client Info are: {single_node_lst}")
    results_one_dct = __run_one_node_simulation(
        data=data,
        node_lst=single_node_lst,
        rounds=rounds,
        file_path=dir_path+'one_node_fl_sim.json'
    )

    #Predict Number of Clients Reduction
    num_clients_pred = __predict_num_nodes(
        train_samples=data['X_train'].shape[0],
        sequence_len=data['X_train'].shape[1],
        classes=tf.keras.utils.to_categorical(data["y_train"]).shape[-1],
        params_count=build_ResNet(data['X_train'].shape[1:], tf.keras.utils.to_categorical(data["y_train"]).shape[-1]).count_params(),
        accuracy_est=results_one_dct[list(results_one_dct.keys())[-1]],
        accuracy_goal=accuracy_goal,
        number_nodes=len(node_options_lst)
    )

    #Apply Recommender Methodology
    if recommender_method == 'NS':
        node_options_sel_lst = nodes_selection(
            node_options_lst=node_options_lst,
            num_clients_pred=num_clients_pred
        )
        #Adjust Data Volume
        node_options_final_lst = __adj_data_volume(
            node_options_lst=node_options_sel_lst,
            num_nodes=len(node_options_lst)
        )
    elif recommender_method == 'SR':
        node_options_sel_lst, percentages = nodes_selection_smart(
            node_options_lst=node_options_lst,
            num_clients_pred=num_clients_pred,
            data=data
        )
        node_options_final_lst = __get_data_volume_req(
            node_options_lst=node_options_sel_lst,
            num_nodes=num_clients_pred,
            percentages=percentages
        )
    elif recommender_method == 'MSR':
        node_options_sel_lst, percetages = nodes_selection_smart_minimal(
            node_options_lst=node_options_lst,
            num_clients_pred=num_clients_pred,
            data=data
        )
        node_options_final_lst = __get_data_volume_req(
            node_options_lst=node_options_sel_lst,
            num_nodes=num_clients_pred,
            percentages=percetages
        )
    else:
        raise ValueError("The 'recommender_method' value inserted is not valid!")
    
    #Run Federated Learning Training Simulation
    results_dct = run_fl_simulation(
        data=data,
        node_options_lst=node_options_final_lst,
        node_selection='fixed',
        fraction_fit=1.0,
        number_clients=num_clients_pred,
        rounds=rounds,
        file_path=dir_path+'recommender_'+recommender_method+'_sim.json',
        recommender=True
    )

    return results_dct
    


def __run_one_node_simulation(
        data: dict,
        node_lst: list,
        rounds: int,
        file_path: str
):
    return run_fl_simulation(
        data=data,
        node_options_lst=node_lst,
        node_selection='fixed',
        fraction_fit=1.0,
        number_clients=1,
        rounds=rounds,
        file_path=file_path
    )

def __predict_num_nodes(
        train_samples: int,
        sequence_len: int,
        classes: int,
        params_count: int,
        accuracy_est: float,
        accuracy_goal: float,
        number_nodes: int
):
    model_input = np.array([[train_samples, sequence_len, classes, params_count, accuracy_est, accuracy_goal]])
    logging.info(f"Model Input: {model_input}")

    #Load Model
    file_model_path = CONFIG['reduction_model_path']
    reduction_model = load(file_model_path)

    #Prediction
    result = reduction_model.predict(model_input)
    num_clients_pred = np.ceil(result[0])
    #Make a proportion with the nodes available, 10 is a costant
    num_clients_pred = int(np.ceil(((num_clients_pred * number_nodes) / 10)))
    if num_clients_pred == 0:
        num_clients_pred = 1
    logging.info(f"Number of nodes predicted: {num_clients_pred}")

    return num_clients_pred

def __adj_data_volume(
        node_options_lst: list,
        num_nodes: int 
):
    '''Compute the required data volume needed per each client'''
    
    total_data_volume_node = round((1.0 / num_nodes), 2)
    for node in node_options_lst:        
        if total_data_volume_node < node['data']['data_volume']['data_quality_dimension_percentage_overall']:
            node['data']['data_volume']['data_quality_dimension_percentage'] = round((total_data_volume_node / node['data']['data_volume']['data_quality_dimension_percentage_overall']), 2)
        else:
            node['data']['data_volume']['data_quality_dimension_percentage'] = 1.0
    
    return node_options_lst

def __get_data_volume_req(
        node_options_lst: list,
        percentages: list,
        num_nodes: int
):
    total_data_volume_node = round((1.0 / num_nodes), 2)
    for index, node in enumerate(node_options_lst):
        #Compute the real total data volume needed per each client
        node['data']['data_volume']['data_quality_dimension_percentage_sr'] = round((percentages[index] / node['data']['data_volume']['data_quality_dimension_percentage_overall']), 2)
    
    return node_options_lst
