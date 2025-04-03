import streamlit as st
import numpy as np
from pipelines.fl_pipeline import run_fl_simulation
from data_processing.load_dataset import load_dataset, splitting_dataset
from frontend.data_visualization import data_visualization
from utils.utils import (
    build_node_options
)


def initialize_session_state():
    """Initialize all session state variables if not already present"""
    if "number_clients_configured" not in st.session_state:
        st.session_state.load_dataset_paths_status = True
        st.session_state.load_partition_status = False
        st.session_state.client_configuration_status = False
        st.session_state.sim_end = False
        st.session_state.number_clients_configured = 0
        st.session_state.data_volume_lst = []
        st.session_state.data_accuracy_lst = []
        st.session_state.data_consistency_lst = []
        st.session_state.data_completeness_lst = []

def configure_dataset():
    """Configure dataset paths and number of clients"""
    st.write("#### Configure Dataset")
    train_dataset_path = st.text_input(
        "Insert Train Set Path",
        "Datasets/Datasets_Training/StarLightsCurves/StarLightCurves_TRAIN.txt",
    )
    test_dataset_path = st.text_input(
        "Insert Test Set Path",
        "Datasets/Datasets_Training/StarLightsCurves/StarLightCurves_TEST.txt",
    )
    clients = st.number_input(
        "Select number of clients of federated configuration:", 1, 100
    )

    if st.button("Apply"):
        st.session_state.dataset_training_path = train_dataset_path
        st.session_state.dataset_test_path = test_dataset_path
        st.session_state.number_clients = clients
        st.session_state.load_dataset_paths_status = False
        st.session_state.load_partition_status = True
        st.rerun()

def load_partition_and_visualization():
    st.write("Scroll Down and Click Continue 👇🏾")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        st.session_state.dataset_training_path, st.session_state.dataset_test_path
    )

    # Split the dataset for federated learning
    indexes_train = splitting_dataset(dataset=X_train, num_clients=st.session_state.number_clients)
    indexes_val = splitting_dataset(dataset=X_val, num_clients=st.session_state.number_clients)

    data_visualization(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        indexes_train=indexes_train,
        indexes_val=indexes_val,
        num_clients=st.session_state.number_clients
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
        st.session_state.client_configuration_status = True
        st.rerun()

def render_client_configuration():
    """Render the client configuration interface"""
    st.write("#### Configure data setting per each node client of your federated learning environment")
    if st.session_state.number_clients_configured > 0:
        st.success(f"{st.session_state.number_clients_configured} Clients already configured! "
                  f"Configure the remaining {st.session_state.number_clients-st.session_state.number_clients_configured} Clients")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_params = {
            'volume': st.slider("Data Volume", 0.00, 1.00),
            'accuracy': st.slider("Data Accuracy", 0.00, 1.00),
            'consistency': st.slider("Data Consistency", 0.00, 1.00),
            'completeness': st.slider("Data Completeness", 0.00, 1.00)
        }

    with col2:
        max_nodes = st.session_state.number_clients - st.session_state.number_clients_configured
        num_nodes = st.number_input("Select the number of nodes to configure:", 1, max_nodes)
        
        if st.button("Apply"):
            st.session_state.client_configuration_status = False
            update_session_state(data_params, num_nodes)
            st.rerun()

def update_session_state(data_params, num_nodes):
    """Update session state with new client configurations"""
    st.session_state.number_clients_configured += num_nodes
    for param_name, param_value in data_params.items():
        getattr(st.session_state, f'data_{param_name}_lst').extend([param_value] * num_nodes)

def render_training_configuration():
    """Render the training configuration interface"""
    node_selection_labels = {
        "Classic": "basic",
        "Fixed": "fixed",
    }

    node_selection = st.radio(
        "## Select how clients will be sampled during the training:",
        list(node_selection_labels.keys()),
        captions=[
            "At the beginning of each round a new set of clients of the same sample size will be taken for training",
            "At the beginning of the first round a sample of clients is chosen to run a federated learning training. The selected set will remain the same for all rounds.",
        ],
    )

    fraction_fit = st.slider("## Percentage of clients used each round for training", 0.00, 1.00)
    rounds = st.number_input("## Select number of rounds for federated training:", 1, 100)
    file_path = st.text_input(
        "Insert json path to save results",
        "Datasets/Datasets_Training/StarLightsCurves/results/results.json"
    )

    return node_selection_labels[node_selection], fraction_fit, rounds, file_path

def run_simulation(node_selection, fraction_fit, rounds, file_path):
    """Execute the federated learning simulation"""
    with st.spinner("Federated Learning Simulation Running...", show_time=True):

        node_options_lst = build_node_options(
            data_volume_lst=st.session_state.data_volume_lst,
            data_accuracy_lst=st.session_state.data_accuracy_lst,
            data_consistency_lst=st.session_state.data_consistency_lst,
            data_completeness_lst=st.session_state.data_completeness_lst,
            num_clients=st.session_state.number_clients
        )

        data_dct = {
            'X_train': st.session_state.data['X_train'],
            'y_train': st.session_state.data['y_train'],
            'X_val': st.session_state.data['X_val'],
            'y_val': st.session_state.data['y_val'],
            'X_test': st.session_state.data['X_test'],
            'y_test': st.session_state.data['y_test'],
            'indexes_train': st.session_state.data['indexes_train'],
            'indexes_val': st.session_state.data['indexes_val'],
        }

        results_dict = run_fl_simulation(
            data=data_dct,
            node_options_lst=node_options_lst,
            node_selection=node_selection,
            fraction_fit=fraction_fit,
            number_clients=st.session_state.number_clients,
            rounds=rounds,
            file_path=file_path
        )

    st.write("The experiment results are:")
    st.json(results_dict)
    st.session_state.sim_end = True


# -------------------------------------------------
# Main --- Streamlit needs global scope executables

st.title("🚀 Federated Learning Simulator")
initialize_session_state()

if st.session_state.load_dataset_paths_status:
    configure_dataset()
elif st.session_state.load_partition_status:
    load_partition_and_visualization()
elif st.session_state.number_clients_configured < st.session_state.number_clients and st.session_state.client_configuration_status:
    render_client_configuration()
else:
    st.success("All Clients have been correctly configured!")
    node_selection, fraction_fit, rounds, file_path = render_training_configuration()
    
    if st.button("Start"):
        run_simulation(node_selection, fraction_fit, rounds, file_path)

if st.session_state.sim_end:
    if st.button("New Simulation", use_container_width=True):
        st.session_state.clear()
        st.rerun()
