import json
import os


def build_node_options(
    data_volume_lst: list[float],
    data_accuracy_lst: list[float],
    data_consistency_lst: list[float],
    data_completeness_lst: list[float],
    num_clients: int,
    recommender_params: dict = {},
    recommender: bool = False
) -> list[dict]:
    """
    The function builds the configuration parameters about data and training for each node

    Args:
        data_volume_lst (list[float]): List of nodes data volume values
        data_accuracy_lst (list[float]): List of nodes data accuracy values
        data_consistency_lst (list[float]): List of nodes data consistency values
        data_completeness_lst (list[float]): List of nodes data completeness values
        num_clients (int): Number of clients in the federated learning configuration

    Return:
        list[dict]: A list containing dictionary options per each node of the federated learning configurations
    """

    if (
        len(data_volume_lst)
        == len(data_accuracy_lst)
        == len(data_consistency_lst)
        == len(data_completeness_lst)
        == num_clients
    ):
        node_options = []
        for i in range(num_clients):
            base_config = {
                    "node_id": i,
                    "data": {
                        "data_volume": {
                            "data_quality_dimension_percentage": 1.0 if recommender else data_volume_lst[i],
                            "data_quality_dimension_percentage_overall": data_volume_lst[i] if recommender else round((1.0 / num_clients), 2),
                            "experiment_method": "uniform",  # Only this kind of experiment method defined for data quality poisoning
                        },
                        "data_accuracy": {
                            "data_quality_dimension_percentage": data_accuracy_lst[i],
                            "experiment_method": "uniform",
                        },
                        "data_consistency": {
                            "data_quality_dimension_percentage": data_consistency_lst[i],
                            "experiment_method": "uniform",
                        },
                        "data_completeness": {
                            "data_quality_dimension_percentage": data_completeness_lst[i],
                            "experiment_method": "uniform",
                        },
                    },
                    "training": {  # Training parameter are currently kept equal per each node
                        "batch_size": 64,
                        "epochs": 40,
                    },
                }
            
            if recommender:
                base_config["name"] = recommender_params["name_lst"][i]
                base_config["location"] = recommender_params["location_lst"][i]
                base_config["energy"] = {
                    "energy_consumption": recommender_params["energy_consumption_lst"][i],
                    "carbon_intensity": recommender_params["carbon_intensity_lst"][i]
                }
            
            node_options.append(base_config)

        return node_options

    else:
        raise ValueError(
            "The dimensions of the lists and the number of clients defined for the federated learning configurations do not correspond!"
        )


def build_results_schema(
    experiment_type: str,
    num_clients_available: int,
    num_clients_trained: int,
    node_options_lst: list = [],
    recommender: bool = False
):

    schema = {
        "experiment_method": experiment_type,
        "node_options": node_options_lst,
        "num_clients_available": num_clients_available,
        "num_clients_trained": num_clients_trained,
        "total_data_volume": 1.0 if recommender else __mean(
            node_options_lst=node_options_lst, dimension="data_volume"
        ),
        "total_data_accuracy": __weighted_mean(
            node_options_lst=node_options_lst, dimension="data_accuracy"
        ) if recommender else __mean(
            node_options_lst=node_options_lst, dimension="data_accuracy"
        ),
        "total_data_consistency": __weighted_mean(
            node_options_lst=node_options_lst, dimension="data_consistency"
        ) if recommender else __mean(
            node_options_lst=node_options_lst, dimension="data_consistency"
        ),
        "total_data_completeness": __weighted_mean(
            node_options_lst=node_options_lst, dimension="data_completeness"
        ) if recommender else __mean(
            node_options_lst=node_options_lst, dimension="data_completeness"
        ),
        "effective_epochs": 0,
        "effective_emissions_kg": 0,
        "effective_energy_consumed": 0,
        "effective_duration": 0,
    }

    return schema


def __mean(node_options_lst: list[dict], dimension: str):
    sum_value = sum(
        node["data"][dimension]["data_quality_dimension_percentage"]
        for node in node_options_lst
    )

    mean = float(format(sum_value / len(node_options_lst), ".2f"))
    return mean

def __weighted_mean(node_options_lst: list[dict], dimension: str):
    sum_value = sum(
        node["data"][dimension]["data_quality_dimension_percentage"]*node["data"]['data_volume']["data_quality_dimension_percentage"]
        for node in node_options_lst
    )

    weighted_mean = float(format(sum_value, ".2f"))
    return weighted_mean

def write_instance_on_results(file_path: str, result_schema: dict):
    if os.path.exists(file_path):
        with open(file_path, "r+") as json_file:
            data = json.load(json_file)
            data.append(result_schema)
            json_file.seek(0)
            json.dump(data, json_file, indent=4)
    else:
        with open(file_path, "w") as json_file:
            json.dump([result_schema], json_file, indent=4)


def read_experiment_results(file_path: str):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data[-1]


def read_main_experiment_results(file_path: str):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    fields = [
        "effective_epochs",
        "effective_emissions_kg",
        "effective_energy_consumed",
        "effective_duration",
        list(data[-1].keys())[-1]
    ]

    results_dict = {key: data[-1][key] for key in fields if key in data[-1]}
    return results_dict