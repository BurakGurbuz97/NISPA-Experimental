from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch
import copy
import os
import csv
import pickle

from NISPA.architecture import Network
from NISPA.train_eval import test
from NISPA.selection import compute_layer_activations
from NISPA.extra_utils import get_data_loaders


def acc_prev_tasks(args: Namespace, task_index: int, scenario: GenericCLScenario, network: Network) -> List[Tuple[str, List]]:
    all_accuracies = []
    for  train_task, val_task, test_task in zip(scenario.train_stream[:task_index+1], 
                                                scenario.val_stream[:task_index+1],  # type: ignore
                                                scenario.test_stream[:task_index+1]):
        task_classes = str(train_task.classes_in_this_experience)
        train_loader, val_loader, test_loader = get_data_loaders(args, train_task, val_task, test_task)
        train_acc =  test(network, train_loader, train_task.classes_in_this_experience)
        val_acc = test(network, val_loader, train_task.classes_in_this_experience)
        test_acc = test(network, test_loader, train_task.classes_in_this_experience)
        all_accuracies.append((task_classes, [train_acc, val_acc, test_acc]))
    return all_accuracies


def _write_units(writer, network: Network):
    weights = network.get_weight_bias_masks_numpy()
    all_units = [list(range(weights[0][0].shape[1]))] + [list(range(w[1].shape[0])) for w in weights]
    # Fix Phase-1 Stable units
    if len(network.list_stable_units):
        stable_units = network.list_stable_units
    else:
        stable_units = [all_units[0]] + [[] for _ in all_units[1:]]

    # Fix Phase-1 Stable and Candidate Stable
    if len(network.current_stable_and_candidate_units):
        stable_and_candidate_units = network.current_stable_and_candidate_units
    else:
        stable_and_candidate_units = [all_units[0]] + [[] for _ in all_units[1:]]

    candidate_stable_units = [list(set(stable_and_candiate).difference(stable_units[i]))
                                for i, stable_and_candiate in enumerate(stable_and_candidate_units)]
    
    plastic_units = [list(set(all_units[i]).difference(stable_and_candidate)) for i, stable_and_candidate in enumerate(stable_and_candidate_units)]

    writer.writerow(["All Units"] + [len(u) for u in all_units])
    writer.writerow(["Stable Units"] + [len(u) for u in stable_units])
    writer.writerow(["Plastic Units"] + [len(u) for u in plastic_units])
    writer.writerow(["Candidate Stable Units"] + [len(u) for u in candidate_stable_units])
    return writer


class PhaseLog():
    def __init__(self, log_params: dict, phase_index: int, task_classes: List, task_index: int):
        self.log_params = log_params
        self.phase_index = phase_index
        self.task_classes = task_classes
        self.task_index = task_index

    def _open_dir(self) -> str:
        dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"],
                              "Task_{}".format(self.task_index), "Phase_{}".format(self.phase_index))
        os.makedirs(dirpath)
        return dirpath

    def archive(self, network: Network, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        if self.log_params["write_phase_log"]:
            dirpath = self._open_dir()
            csvfile = open(os.path.join(dirpath, "Phase_{}.csv".format(self.phase_index)), 'w', newline='')
            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)
            
            if self.log_params["eval_model_phase"]:
                writer.writerow(["Train Accuracy", test(network, train_loader, self.task_classes)])
                writer.writerow(["Validation Accuracy", test(network, val_loader, self.task_classes)])
                writer.writerow(["Test Accuracy", test(network, test_loader, self.task_classes)])
            csvfile.close()

            if self.log_params["save_activations_phases"]:
                _, average_activations_on_train = compute_layer_activations(network, train_loader)
                with open(os.path.join(dirpath, 'average_activations_on_train_samples.pkl'), 'wb') as file:
                    pickle.dump(average_activations_on_train, file)
            if self.log_params["save_model_phase"]:
                torch.save(network.state_dict(), os.path.join(dirpath, 'network_end_of_phase.pth'))
           

class TaskLog():
    def __init__(self, args: Namespace, log_params: dict, task_index: int, scenario: GenericCLScenario):
        self.log_params = log_params
        self.scenario = scenario
        self.task_index = task_index
        self.args = args

    def archive(self, network: Network):
        if self.log_params["write_task_log"]:
            dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"],
                              "Task_{}".format(self.task_index))
            csvfile = open(os.path.join(dirpath, "Task_{}.csv".format(self.task_index)), 'w', newline='')

            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)
            # This function assumes task_index starts from 0 so we have -1
            prev_task_accs = acc_prev_tasks(self.args, self.task_index - 1, self.scenario, network)
            for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
                writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc), "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])

            if self.log_params["save_activations_task"]:
                # This function assumes task_index starts from 0 so we have -1
                train_task = self.scenario.train_stream[self.task_index - 1]
                # This creates three loaders TODO: refactor
                train_loader, _, _ = get_data_loaders(self.args, train_task, train_task, train_task)
                _, average_activations_on_train = compute_layer_activations(network, train_loader)
                with open(os.path.join(dirpath, 'average_activations_on_train_samples.pkl'), 'wb') as file:
                    pickle.dump(average_activations_on_train, file)
            if self.log_params["save_model_task"]:
                torch.save(network.state_dict(), os.path.join(dirpath, 'network_end_of_phase.pth'))
            csvfile.close()

class SequenceLog():
    def __init__(self, args: Namespace, log_params: dict, scenario: GenericCLScenario):
        self.log_params = log_params
        self.scenario = scenario
        self.args = args

    def archive(self, network: Network):
        if self.log_params["write_sequence_log"]:
            dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"])
            csvfile = open(os.path.join(dirpath, "End_of_Sequence.csv"), 'w', newline='')
            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)
            # This function assumes task_index starts from 0 so we have -1
            prev_task_accs = acc_prev_tasks(self.args, self.args.number_of_tasks - 1, self.scenario, network)
            for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
                writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc), "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])

            torch.save(network.state_dict(), os.path.join(dirpath, 'network_final.pth'))
            csvfile.close()