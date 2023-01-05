from typing import List, Tuple
from numpy import typing as np_type
import numpy as np
import torch
import copy

from NISPA.architecture import Network
from NISPA.model_utils import MaskedConv2dDynamic, MaskedLinearDynamic, get_device
from NISPA.extra_utils import get_plastic_units


def drop_connections_plastic_to_stable(stable_and_candidate_stable_units: List[List[int]], network: Network) -> Tuple[Network, List[int]]:
    def create_drop_masks(weights: List[Tuple[np_type.NDArray[np.double], np_type.NDArray[np.double]]],
                       stable_and_candidate_stable_units: List[List[int]], plastic_units: List[List[int]]) -> List[np_type.NDArray[np.intc]]:
        drop_masks = []
        for i, (prev_layer_plastic_indices, next_layer_stable_candidate_indices) in enumerate(zip(plastic_units[:-1],
                                                                                                    stable_and_candidate_stable_units[1:])):
            mask_drop = np.zeros(weights[i][0].shape, dtype=np.intc)
            if prev_layer_plastic_indices:
                #Conv2Linear
                if len(weights[i][0].shape) == 2 and len(weights[i-1][0].shape) == 4:
                    for plastic_index in prev_layer_plastic_indices:
                        start = plastic_index*network.config_dict["conv"]["conv2lin_mapping_size"]
                        end = (plastic_index+1)*network.config_dict["conv"]["conv2lin_mapping_size"]
                        mask_drop[next_layer_stable_candidate_indices, start:end] = 1
                else:
                    mask_drop[np.ix_(next_layer_stable_candidate_indices, prev_layer_plastic_indices)] = 1
            drop_masks.append(mask_drop * weights[i][0])
        return drop_masks

    def drop_connections_with_mask(network: Network, drop_masks: List[np_type.NDArray[np.intc]]) -> Tuple[Network, List[int]]:
        mask_index, num_drops = 0, []
        for module in network.modules():
            if isinstance(module, MaskedLinearDynamic) or isinstance(module, MaskedConv2dDynamic):
                weight_mask, bias_mask = module.get_mask()
                weight_mask[torch.tensor(drop_masks[mask_index], dtype= torch.bool)] = 0
                num_drops.append(int(np.sum(drop_masks[mask_index])))
                module.set_mask(weight_mask, bias_mask) # type: ignore
                mask_index += 1
        return network, num_drops

    weights = network.get_weight_bias_masks_numpy() 
    plastic_units = get_plastic_units(network, stable_and_candidate_stable_units)
    drop_masks = create_drop_masks(weights, stable_and_candidate_stable_units, plastic_units)
    return drop_connections_with_mask(network, drop_masks)


def fix_plastic_units_without_outgoing(network: Network, stable_and_candidate_stable_units: List[List[int]], connection_quota: List[int]) -> Tuple[Network, List[int]]:
    weights = network.get_weight_bias_masks_numpy() 
    all_plastic_units = get_plastic_units(network, stable_and_candidate_stable_units)
    new_masks = [weights[0]]
    for i, plastic_units in enumerate(all_plastic_units[1:-1], 1):
        layer_mask = weights[i][0]
        kernel_size = (layer_mask.shape[2] * layer_mask.shape[3]) if len(layer_mask.shape) == 4 else 1
        for plastic_unit in plastic_units:
            # not enough connections to fix
            if connection_quota[i] - kernel_size < 0:
                continue
            if len(all_plastic_units[i + 1]) == 0:
                print("Warning: layer-{} is fully stable, we have plastic units without outgoing connections at layer-{}.".format(i+1, i))
                continue
            # plastic_unit has no outgoing connections
            if np.sum(layer_mask[:, plastic_unit]) == 0:
                target = np.random.choice(all_plastic_units[i + 1], 1)
                layer_mask[target, plastic_unit] = 1
                connection_quota[i] = connection_quota[i] - kernel_size
        new_masks.append((layer_mask, weights[i][1])) # we do not need to modify bias
    new_masks.append((weights[-1][0], weights[-1][1]))
    w_masks, b_masks = [torch.tensor(w_mask) for w_mask, _ in new_masks], [torch.tensor(b_mask) for _, b_mask in new_masks]
    network.set_masks(w_masks, b_masks)
    return network, connection_quota


class Growth():
    def __init__(self, stable_units: List[List[int]], candidate_stable_units: List[List[int]], plastic_units: List[List[int]]):
        self.stable_units = stable_units
        self.candidate_units = candidate_stable_units
        self.plastic_units = plastic_units

    def grow(self, network: Network, connection_quota: List[int], grow_algo = "random", grow_init = "zero_init") -> Tuple[Network, List[int]]:
        # No quota
        if sum(connection_quota) == 0:
            return network, connection_quota
        
        if grow_algo == "random":
            return self._random_grow(network, connection_quota, grow_init)
        raise Exception("Grow mechanism {} is not supported.".format(grow_algo))

    # ------- Helper methods ------- # 
    def _grow_connections(self, network: Network, possible_connections: List[np_type.NDArray[np.bool8]],
                              connection_probs: List[np_type.NDArray[np.float64]], connection_quota: List[int], grow_init) -> Tuple[Network, List[int]]:
        layer_index = 0
        weight_init = getattr(self, "_" + grow_init)
        remainder_connections = []
        for module in network.modules():
            if isinstance(module, MaskedLinearDynamic) or isinstance(module, MaskedConv2dDynamic):
                if connection_quota[layer_index] == 0:
                    remainder_connections.append(0)
                    layer_index = layer_index + 1
                    continue
                weight_mask, bias_mask = module.get_mask()
                # Conv layer
                if len(possible_connections[layer_index].shape) == 4:
                    grow_indices = np.nonzero(np.sum(possible_connections[layer_index], axis = (2 , 3)))
                # Linear layer
                else:
                    grow_indices = np.nonzero(possible_connections[layer_index])
                
                conn_shape = (possible_connections[layer_index].shape[2], possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else (1,)
                conn_size = (possible_connections[layer_index].shape[2] * possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else 1
                probs = connection_probs[layer_index][grow_indices]

                # There are connections that we can grow
                if len(grow_indices[0]) != 0:
                    # We can partial accommodate grow request (we will have remainder connections)
                    if len(grow_indices[0])*conn_size <= connection_quota[layer_index]:
                        weight_mask[grow_indices] = 1
                        module.weight.data[grow_indices] = weight_init(module, conn_size)
                        remainder_connections.append(connection_quota[layer_index] - len(grow_indices[0])*conn_size)
                    else:
                        # Select based on probability
                        try:
                            selection = np.random.choice(len(grow_indices[0]), size = int(connection_quota[layer_index]/ conn_size), replace = False, p = probs)
                        except:
                            selection = np.random.choice(len(grow_indices[0]), size = int(connection_quota[layer_index]/ conn_size), replace = False)
                            print("Warning: Not enough possible connections to sample properly! Layer: ",layer_index)
                        
                        tgt_selection = torch.tensor(grow_indices[0][selection]).to(get_device())
                        src_selection = torch.tensor(grow_indices[1][selection]).to(get_device())
                        weight_mask[tgt_selection, src_selection] = torch.squeeze(torch.ones((len(tgt_selection), *conn_shape), dtype = weight_mask.dtype)).to(get_device())
                        module.weight.data[tgt_selection, src_selection] = torch.squeeze(weight_init(module, (len(tgt_selection), *conn_shape)))
                        remainder_connections.append(0)
                else:
                    remainder_connections.append(connection_quota[layer_index])
                module.set_mask(weight_mask, bias_mask)  # type: ignore
                layer_index += 1
        return network, remainder_connections
    # ------- Connection growth methods ------- #
    def _random_grow(self, network: Network, connection_quota: List[int], grow_init) -> Tuple[Network, List[int]]:

        def get_possible_conns_and_probs(network: Network, plastic_units: List[List[int]]):
            weights = network.get_weight_bias_masks_numpy()
            possible_connections = []
            connection_probs = []
            for layer_index, plastic_targets in enumerate(plastic_units[1:]):
                if len(plastic_targets) == 0:
                    pos_conn = np.zeros(weights[layer_index][0].shape)
                    prob_matrix = copy.deepcopy(pos_conn.sum(axis = (2, 3))) if len(weights[layer_index][0].shape) == 4 else copy.deepcopy(pos_conn)
                    possible_connections.append(pos_conn)
                    connection_probs.append(prob_matrix)
                    continue
                conn_type_1 = np.ones((weights[layer_index][0].shape[2], weights[layer_index][0].shape[3])) if len(weights[layer_index][0].shape) == 4 else 1
                conn_type_0 = np.zeros((weights[layer_index][0].shape[2], weights[layer_index][0].shape[3])) if len(weights[layer_index][0].shape) == 4 else 0
                pos_conn = np.zeros(weights[layer_index][0].shape)
                pos_conn[plastic_targets,:] = conn_type_1
                if len(weights[layer_index][0].shape) == 4:
                    pos_conn[np.all(weights[layer_index][0][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
                else:
                    pos_conn[weights[layer_index][0] != 0] = 0
                if len(weights[layer_index][0].shape) == 4:
                    prob_matrix = copy.deepcopy(pos_conn.sum(axis = (2, 3)))
                else:
                    prob_matrix = copy.deepcopy(pos_conn)
                prob_matrix[np.nonzero(prob_matrix)] =  prob_matrix[np.nonzero(prob_matrix)] / np.sum(prob_matrix)
                possible_connections.append(pos_conn)
                connection_probs.append(prob_matrix)
            return possible_connections, connection_probs

        possible_connections, connection_probs = get_possible_conns_and_probs(network, self.plastic_units)
        return self._grow_connections(network, possible_connections, connection_probs, connection_quota, grow_init)

    #  ------- Weight initialization methods ------- #
    @staticmethod
    def _zero_init(module: torch.nn.Module, size: Tuple) -> torch.Tensor:
        return torch.zeros(size).to(get_device())


