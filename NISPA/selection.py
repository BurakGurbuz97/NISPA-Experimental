from argparse import Namespace
import math
import numpy as np
from numpy import typing as np_type
import torch
from typing import Callable, List, Tuple
from torch.utils.data import DataLoader

from  NISPA.architecture import Network, get_device

def get_tau_schedule(args: Namespace) -> Callable[..., float]:

    class funcs():
        def __init__(self, k):
            self.cosine_anneling = lambda t: 0.5 * (1 + math.cos(t * math.pi / k))
            self.linear = lambda t: 1 - k*t
            self.exp_decay = lambda t: (t + 1)**(-k)

    return getattr(funcs(args.tau_param), args.tau_schedule)


def remove_units_without_incoming(units_of_interest: List[List[int]], network: Network) -> List[List[int]]:
    weights = network.get_weight_bias_masks_numpy()
    new_units = [units_of_interest[0]]
    for i, (source, target) in enumerate(zip(units_of_interest[:-1], units_of_interest[1:])):
        source, target = np.array(source, dtype=np.int32), np.array(target, dtype=np.int32)
        dead_units = []
        for tgt in target:
            if np.sum(np.abs(weights[i][0][tgt, source])) == 0:
                dead_units.append(tgt)
        if len(dead_units) != 0:
            updated_units = list(target)
            for dead_unit in dead_units:
                updated_units.remove(dead_unit)
            new_units.append(updated_units)
        else:
            new_units.append(list(target))
    return new_units


def compute_layer_activations(network: Network, train_loader: DataLoader) -> Tuple[List[np_type.NDArray[np.double]], List[np_type.NDArray[np.double]]]:
    total_activations = []
    network.train()
    with torch.no_grad():
         for data, _, _ in train_loader:
            data = data.to(get_device())
            activations = [activation.detach().cpu().numpy() for activation in network.forward_activations(data)]
            batch_sum_activation = [np.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else  np.sum(activation, axis = 0)
                                    for activation in activations]  
            total_activations =  [total_activations[i]+activation for i, activation in enumerate(batch_sum_activation)] if total_activations else batch_sum_activation

    average_activations = [total_activation/len(train_loader.dataset) for total_activation in total_activations]  # type: ignore
    return total_activations, average_activations

def select_candidate_stable_units(network: Network, train_loader: DataLoader, stable_selection_perc: float) -> Tuple[List[List[int]],  List[List[int]]]:

    def pick_top_neurons(average_layer_activation: np_type.NDArray[np.float32], stable_selection_perc: float) -> List[int]:
        total = sum(average_layer_activation)
        accumulate = 0
        indices = []
        sort_indices = np.argsort(-average_layer_activation)
        for index in sort_indices:
            accumulate = accumulate + average_layer_activation[index]
            indices.append(index)
            if accumulate >= total * stable_selection_perc / 100:
                break
        return indices
        
    _, average_activations = compute_layer_activations(network, train_loader)
    selected_candidate_units = []
    for average_layer_activation in average_activations[:-1]:
        selected_candidate_units.append(pick_top_neurons(average_layer_activation, stable_selection_perc))
    
    # Add input and output
    selected_candidate_units = [list(range(network.input_size))] + selected_candidate_units + [network.classes_seen_so_far]
    
    # No stable units
    if len(network.list_stable_units) == 0:
        stable_and_candidate_stable_units = selected_candidate_units
    # We have stable units already
    else:
        stable_and_candidate_stable_units = [list(set(candidate_stable_units).union(stable_units)) 
                                            for candidate_stable_units, stable_units in zip(selected_candidate_units, network.list_stable_units)]

    return stable_and_candidate_stable_units, selected_candidate_units

