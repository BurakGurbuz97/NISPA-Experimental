from torch.utils.data import DataLoader
from typing import Tuple
from argparse import Namespace
from avalanche.benchmarks import TCLExperience
from typing import List
from torch import nn

from .architecture import Network


class PhaseVars():
    def __init__(self, pruned_network: Network):
        self.phase_index = 1
        self.previous_model = pruned_network
        self.best_phase_acc = 0.0
        self.prev_phase_stable_and_candidate_stable_units = []
        self.stable_and_candidate_stable_units = pruned_network.list_stable_units

def get_loss(args: Namespace) -> nn.Module:
    return getattr(nn, args.loss_func)
    

def get_plastic_units(network: Network, stable_and_candidate_stable_units: List[List[int]]) -> List[List[int]]:
    weights = network.get_weight_bias_masks_numpy()
    units = [w[1].shape[0] for w in weights]
    plastic_units = []
    for i,  stable_and_candidate_unit in enumerate(stable_and_candidate_stable_units[1:]):
        all_units = set(range(units[i]))
        plastic_units.append(list(all_units.difference(set(stable_and_candidate_unit))))
    return [[]] + plastic_units # Inputs are always stable


def get_data_loaders(args: Namespace, train_task: TCLExperience, val_task: TCLExperience, test_task: TCLExperience) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_task.dataset, batch_size = args.batch_size,  shuffle=True)
    val_loader =  DataLoader(val_task.dataset, batch_size = args.batch_size,  shuffle=True)
    test_loader = DataLoader(test_task.dataset, batch_size = args.batch_size,  shuffle=True)
    return (train_loader, val_loader, test_loader)
