import streamlit as st
import numpy as np
import tensorflow as tf
from utils.utils import (
    build_node_options
)
from data_processing.load_dataset import load_dataset, splitting_dataset
from frontend.data_visualization import data_visualization
from pipelines.rec_pipeline import run_fl_recommender
from pipelines.fl_pipeline import run_fl_simulation


def initialize_session_state():
    """Initialize all session state variables if not already present"""
    st.session_state.initialization_status = True
    st.session_state.check_status = False
    st.session_state.load_partition_status = False
    st.session_state.load_dataset_paths_status = True
    st.session_state.option_selection_status = False
    st.session_state.full_mode_status = False
    st.session_state.rec_mode_status = False
    st.session_state.sim_end = False
    st.session_state.number_clients = 0
    st.session_state.number_clients_configured = 0
    st.session_state.data_volume_lst = []
    st.session_state.data_accuracy_lst = []
    st.session_state.data_consistency_lst = []
    st.session_state.data_completeness_lst = []
    st.session_state.rec_params = {
        "name_lst": [],
        "energy_consumption_lst": [],
        "location_lst": [],
        "carbon_intensity_lst": []
    }

def configure_dataset():
    """Configure dataset paths and number of clients"""
    st.write("#### Configure Dataset")
    train_dataset_path = st.text_input(
        "Insert Train Set Path",
    )
    test_dataset_path = st.text_input(
        "Insert Test Set Path",
    )
    clients = st.number_input(
        "Select number of clients of federated configuration:", 1, 100
    )

    if st.button("Apply"):
        st.session_state.dataset_training_path = train_dataset_path
        st.session_state.dataset_test_path = test_dataset_path
        st.session_state.number_clients = clients
        st.session_state.load_dataset_paths_status = False
        st.rerun()

def render_client_configuration():
    """Render the client configuration interface with energy metrics"""
    st.write("#### Configure data setting per each Node Client!")
    if st.session_state.number_clients_configured > 0:
        st.success(f"{st.session_state.number_clients_configured} Clients already configured! "
                  f"Configure the remaining {st.session_state.number_clients-st.session_state.number_clients_configured} Clients")
    
    col1, col2, col3 = st.columns([2,1,2])
    
    with col1:
        node_name = st.text_input("Node Name:")
        st.divider()
        st.write("**Insert Data Node Characteristics:**")
        data_params = {
            'volume': st.slider("Data Volume", 0.00, 1.00),
            'accuracy': st.slider("Data Accuracy", 0.00, 1.00),
            'consistency': st.slider("Data Consistency", 0.00, 1.00),
            'completeness': st.slider("Data Completeness", 0.00, 1.00)
        }

    with col3:
        st.write("**Insert Node Energy Mix:**")
        energy_params = {
            'consumption': st.slider("Energy Consumption (kWh)", 0.00, 1000.00),
            'location': st.text_input("Location:"),
            'carbon_intensity': st.slider("Carbon Intensity (gCO2eq/kWh)", 0.00, 10000.00)
        }
        st.divider()

        max_nodes = st.session_state.number_clients - st.session_state.number_clients_configured
        num_nodes = st.number_input("Select the number of nodes to configure:", 1, max_nodes)
        
        if st.button("**Apply**"):
            update_session_state(data_params, energy_params, node_name, num_nodes)
            if st.session_state.number_clients_configured == st.session_state.number_clients and st.session_state.number_clients > 0:
                st.session_state.check_status = True
            st.rerun()

def update_session_state(data_params, energy_params, node_name, num_nodes):
    """Update session state with new client configurations including energy metrics"""
    st.session_state.number_clients_configured += num_nodes
    for param_name, param_value in data_params.items():
        getattr(st.session_state, f'data_{param_name}_lst').extend([param_value] * num_nodes)
    
    st.session_state.rec_params['name_lst'].extend([node_name] * num_nodes)
    st.session_state.rec_params['energy_consumption_lst'].extend([energy_params['consumption']] * num_nodes)
    st.session_state.rec_params['location_lst'].extend([energy_params['location']] * num_nodes)
    st.session_state.rec_params['carbon_intensity_lst'].extend([energy_params['carbon_intensity']] * num_nodes)


def check_volume():
    if sum(st.session_state.data_volume_lst) != 1.0:
        st.error("The Sum of the Node Data Volumes must be equal to 1.0")

        restart_button = st.button("**Restart**", use_container_width=True)
        if restart_button:
            st.session_state.clear()
            st.rerun()
    
    else:
        st.session_state.check_status = False
        st.session_state.load_partition_status = True
        st.rerun()

def load_partition_and_visualization():
    st.write("Scroll Down and Click Continue 👇🏾")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        st.session_state.dataset_training_path, st.session_state.dataset_test_path
    )

    # Split the dataset for federated learning
    indexes_train = splitting_dataset(dataset=X_train, num_clients=st.session_state.number_clients, data_volume_lst=st.session_state.data_volume_lst, recommender=True)
    indexes_val = splitting_dataset(dataset=X_val, num_clients=st.session_state.number_clients, data_volume_lst=st.session_state.data_volume_lst, recommender=True)

    data_visualization(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        indexes_train=indexes_train,
        indexes_val=indexes_val,
        num_clients=st.session_state.number_clients,
        num_clients_names= st.session_state.rec_params['name_lst']
    )

    continue_button = st.button("**Continue**", use_container_width=True)
    if continue_button:
        st.session_state.data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'indexes_train': indexes_train,
            'indexes_val': indexes_val
        }
        st.session_state.load_partition_status = False
        st.session_state.option_selection_status = True
        st.rerun()

def option_selection():
    option_labels = {
        "Full Resource": "full",
        "Recommendation System": "rec",
    }

    strategy_labels = {
        "Node Selection": "NS",
        "Smart Reduction": "SR",
        "Minimal Smart Reduction": "MSR"
    }

    option_selection = st.radio(
        "## How do you want to procede?",
        list(option_labels.keys()),
        captions=[
            "Training with all nodes and all available data. Select it for getting baseline results",
            "Training through the Recommendation System and Data Reduction"
        ],
    )

    st.divider()

    if option_labels[option_selection] == "rec":
        strategy_selection = st.radio(
            "## Choose the **Recommender Methodology** you would like to apply:",
            list(strategy_labels.keys()),
        )
        st.session_state.accuracy_goal = st.slider("Accuracy Goal:", 0.00, 1.00)

    st.session_state.n_rounds = st.number_input("Insert the Number of Rounds:", 1, 100)
    st.session_state.dir_path = st.text_input("Insert Directory Path to Save Results:")

    continue_button = st.button("**Continue**", use_container_width=True)
    if continue_button:
        st.session_state.option_selection_status = False
        if option_labels[option_selection] == "full":
            st.session_state.full_mode_status = True
        
        if option_labels[option_selection] == "rec":
            st.session_state.rec_method = strategy_labels[strategy_selection]
            st.session_state.rec_mode_status = True

        st.rerun()


def rec_pipeline():
    with st.spinner("Simulation is running. Check out the logs on terminal!", show_time=True):
        node_options_lst = build_node_options(
            data_volume_lst=st.session_state.data_volume_lst,
            data_accuracy_lst=st.session_state.data_accuracy_lst,
            data_consistency_lst=st.session_state.data_consistency_lst,
            data_completeness_lst=st.session_state.data_completeness_lst,
            num_clients=st.session_state.number_clients,
            recommender_params=st.session_state.rec_params,
            recommender=True
        )

        results_dct = run_fl_recommender(
            data=st.session_state.data,
            node_options_lst=node_options_lst,
            recommender_method=st.session_state.rec_method,
            accuracy_goal=st.session_state.accuracy_goal,
            rounds=st.session_state.n_rounds,
            dir_path=st.session_state.dir_path
        )

        st.json(results_dct)
        st.session_state.rec_mode_status = False
        st.session_state.sim_end = True

def full_resource_pipeline():
    with st.spinner("Simulation is running. Check out the logs on terminal!", show_time=True):
        node_options_lst = build_node_options(
                data_volume_lst=st.session_state.data_volume_lst,
                data_accuracy_lst=st.session_state.data_accuracy_lst,
                data_consistency_lst=st.session_state.data_consistency_lst,
                data_completeness_lst=st.session_state.data_completeness_lst,
                num_clients=st.session_state.number_clients,
                recommender_params=st.session_state.rec_params,
                recommender=True
            )
        
        results_dct = run_fl_simulation(
            data=st.session_state.data,
            node_options_lst=node_options_lst,
            node_selection='fixed',
            fraction_fit=1.0,
            number_clients=st.session_state.number_clients,
            rounds=st.session_state.n_rounds,
            file_path=st.session_state.dir_path+'full_resource_sim.json'
        )
        
        st.json(results_dct)
        st.session_state.full_mode_status = False
        st.session_state.sim_end = True
        


# -------------------------------------------------
# Main --- Streamlit needs global scope executables
st.title("🔮 Federated Learning Recommender")

if 'initialization_status' not in st.session_state:
    initialize_session_state()

if st.session_state.load_dataset_paths_status:
    configure_dataset()
elif st.session_state.number_clients_configured < st.session_state.number_clients:
    render_client_configuration()
elif st.session_state.check_status:
    check_volume()
elif st.session_state.load_partition_status:
    load_partition_and_visualization()
elif st.session_state.option_selection_status:
    option_selection()
elif st.session_state.full_mode_status:
    full_resource_pipeline()
elif st.session_state.rec_mode_status:
    rec_pipeline()
else:
    pass

if st.session_state.sim_end:
    if st.button("**New Simulation**", use_container_width=True):
        st.session_state.clear()
        st.rerun()