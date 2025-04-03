from data_processing.data_quality import *


# We rank the nodes based on a score resulting from weighting the emission and the quality of the data. The volume of the data is analyzed after
def ranking_nodes(single_node, max_emi, weight_emi, weight_cons, weight_com):
    emissions = round(
        single_node["energy"]["carbon_intensity"] * single_node["energy"]["energy_consumption"], 2
    )  # Quantity of emission per hour
    score = (
        (weight_emi * (1 - (emissions / max_emi)))
        + (weight_cons * single_node["data"]["data_consistency"]["data_quality_dimension_percentage"])
        + (weight_com * single_node["data"]["data_completeness"]["data_quality_dimension_percentage"])
    )

    return score


# Node Selection Method
def nodes_selection(
    node_options_lst: list,
    num_clients_pred: int,
    weight_emission: float = 0.7,
    weight_consistency: float = 0.2,
    weight_completeness: float = 0.1
):

    # Total number of clients available
    n_total_clients = len(node_options_lst)

    # Weight are normalized
    weight = weight_emission + weight_consistency + weight_completeness
    # Take the maximum to normalize the emissions
    emissions = []
    for node in node_options_lst:
        emissions.append(
            round(node['energy']["carbon_intensity"] * node['energy']["energy_consumption"], 2)
        )
    max_emission = max(emissions)
    if weight <= 1.0:
        data_sorted = sorted(
            node_options_lst,
            key=lambda x: ranking_nodes(
                x,
                max_emission,
                weight_emission,
                weight_consistency,
                weight_completeness,
            ),
            reverse=True,
        )

    # Data_volume per each client requested
    data_volume_node = round((1.0 / n_total_clients), 2)

    # Check if a node has the data_volume requested otherwise do not select it and select the next node
    nodes_data_suc = []
    nodes_data_last_suc = []
    for obj_exp in data_sorted:
        if data_volume_node < obj_exp["data"]["data_volume"]["data_quality_dimension_percentage"]:
            nodes_data_suc.append(obj_exp)
        else:
            nodes_data_last_suc.append(obj_exp)

    # If a node has not enough data samples, it will be added at the end
    nodes_data_suc.extend(nodes_data_last_suc)

    # This is the list of the selected nodes
    nodes = nodes_data_suc[:num_clients_pred]

    return nodes


# Smart Reduction Method
def nodes_selection_smart(
    node_options_lst: list,
    num_clients_pred: int,
    data: dict,
    weight_emission: float = 0.7,
    weight_consistency: float = 0.2,
    weight_completeness: float = 0.1
):

    # Total number of clients available
    n_total_clients = len(node_options_lst)

    # Weight are normalized
    weight = weight_emission + weight_consistency + weight_completeness
    # Take the maximum to normalize the emissions
    emissions = []
    for node in node_options_lst:
        emissions.append(
            round(node["energy"]["carbon_intensity"] * node["energy"]["energy_consumption"], 2)
        )
    max_emission = max(emissions)
    if weight <= 1.0:
        data_sorted = sorted(
            node_options_lst,
            key=lambda x: ranking_nodes(
                x,
                max_emission,
                weight_emission,
                weight_consistency,
                weight_completeness,
            ),
            reverse=True,
        )

    # Data_volume per each client requested
    data_volume_node = round((1.0 / n_total_clients), 2)
    # Total data volume requested
    data_volume_total = round((data_volume_node * num_clients_pred), 2)

    # Check if a node has the data_volume requested otherwise do not select it and select the next node
    nodes_data_suc = []
    nodes_data_last_suc = []
    for obj_exp in data_sorted:
        if data_volume_node < obj_exp["data"]["data_volume"]["data_quality_dimension_percentage_overall"]:
            nodes_data_suc.append(obj_exp)
        else:
            nodes_data_last_suc.append(obj_exp)

    # If a node has not enough data samples, it will be added at the end
    nodes_data_suc.extend(nodes_data_last_suc)

    # List of the selected nodes
    active_nodes = nodes_data_suc[:num_clients_pred]
    # List of future candidates
    candidates = nodes_data_suc[num_clients_pred:]
    print(f"Number active nodes {len(active_nodes)}")

    data_volume_sum = 0
    # I want to retrieve the exact percentages per each node
    percentages = []
    # nodes_to_be_removed = []
    # Smart Reduction of the datasets
    for node in active_nodes:
        final_percentage = compute_effective_percentage(
            node, data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['indexes_train'], data['indexes_val']
        )
        print(f"Final percentage of node {node['node_id']} is {final_percentage}")
        if data_volume_sum < data_volume_total and final_percentage > 0.0:
            data_volume_sum += final_percentage
            percentages.append(final_percentage)

    # Remove the items with an inappropiate data volume
    # active_nodes = [item for item in active_nodes if item not in nodes_to_be_removed]
    active_nodes = active_nodes[: len(percentages)]

    # Select other nodes to fulfill the total data volume requested
    i = 0
    candidates_to_be_removed = []
    print(f"Data volume totale: {data_volume_total}")
    while data_volume_sum < data_volume_total:
        print(f"Data volume sum: {data_volume_sum}, {i}, {len(active_nodes)}")
        obj = candidates[i]
        if (data_volume_total - data_volume_sum) < obj["data"]["data_volume"]["data_quality_dimension_percentage_overall"]:
            final_percentage_second = compute_effective_percentage(
                obj, data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['indexes_train'], data['indexes_val']
            )
            if final_percentage_second > 0.00:
                print(f"Final percentage: {final_percentage_second}")
                active_nodes.append(obj)
                candidates_to_be_removed.append(obj)
                data_volume_sum += final_percentage_second
                percentages.append(final_percentage_second)
        i += 1
        if i == len(candidates):
            break

    candidates = [item for item in candidates if item not in candidates_to_be_removed]

    i = 0
    # If no the condition is not satisfied do it without (data_volume_total - data_volume_sum) < obj["data_volume"]:
    while data_volume_sum < data_volume_total:
        print(f"Data volume sum: {data_volume_sum}, {i}, {len(active_nodes)}")
        obj = candidates[i]
        final_percentage_second = compute_effective_percentage(
            obj, data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['indexes_train'], data['indexes_val']
        )
        if final_percentage_second > 0.00:
            print(f"Final percentage: {final_percentage_second}")
            active_nodes.append(obj)
            data_volume_sum += final_percentage_second
            percentages.append(final_percentage_second)
        i += 1
        if i == len(candidates):
            break

    return active_nodes, percentages


# Minimal Smart Reduction Method
def nodes_selection_smart_minimal(
    node_options_lst: list,
    num_clients_pred: int,
    data: dict,
    weight_emission: float = 0.7,
    weight_consistency: float = 0.2,
    weight_completeness: float = 0.1
):

    # Total number of clients available
    n_total_clients = len(node_options_lst)

    # Weight are normalized
    weight = weight_emission + weight_consistency + weight_completeness
    # Take the maximum to normalize the emissions
    emissions = []
    for node in node_options_lst:
        emissions.append(
            round(node["energy"]["carbon_intensity"] * node["energy"]["energy_consumption"], 2)
        )
    max_emission = max(emissions)
    if weight <= 1.0:
        data_sorted = sorted(
            node_options_lst,
            key=lambda x: ranking_nodes(
                x,
                max_emission,
                weight_emission,
                weight_consistency,
                weight_completeness,
            ),
            reverse=True,
        )

    # Data_volume per each client requested
    data_volume_node = round((1.0 / n_total_clients), 2)

    # Check if a node has the data_volume requested otherwise do not select it and select the next node
    nodes_data_suc = []
    nodes_data_last_suc = []
    for obj_exp in data_sorted:
        if data_volume_node < obj_exp["data"]["data_volume"]["data_quality_dimension_percentage_overall"]:
            nodes_data_suc.append(obj_exp)
        else:
            nodes_data_last_suc.append(obj_exp)

    # If a node has not enough data samples, it will be added at the end
    nodes_data_suc.extend(nodes_data_last_suc)

    # List of the selected nodes
    active_nodes = nodes_data_suc[:num_clients_pred]
    print(f"Number active nodes {len(active_nodes)}")

    percentages = []
    ultimate_nodes = []
    for node in active_nodes:
        percentage = compute_effective_percentage(
            node, data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['indexes_train'], data['indexes_val']
        )
        if percentage > 0.0:
            percentages.append(percentage)
            ultimate_nodes.append(node)
        else:
            percentage = compute_effective_percentage(
                nodes_data_suc[num_clients_pred], data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['indexes_train'], data['indexes_val']
            )
            percentages.append(percentage)
            ultimate_nodes.append(nodes_data_suc[num_clients_pred])

    return ultimate_nodes, percentages
