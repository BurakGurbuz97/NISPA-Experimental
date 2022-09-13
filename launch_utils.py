import argparse
import config
from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitFMNIST, SplitMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream
import model_config
import numpy as np
import torch
import random
from typing import Tuple
from torch.backends import cudnn 
from torchvision import transforms
import os
import pickle
import shutil

def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='NISPA')
    # Logging params
    parser.add_argument('--experiment_name', type=str, default = 'Test')
    parser.add_argument('--experiment_note', type=str, default = '')

    # Dataset params
    parser.add_argument('--dataset', type=str, default = 'SplitMNIST')
    parser.add_argument('--number_of_tasks', type=int, default = 5)
    parser.add_argument('--CIL_setting', type=str, default = 'TIL', choices=['TIL'])

    # Architectural params
    parser.add_argument('--model', type=str, default = 'Mlp_Vanilla')
    parser.add_argument('--prune_perc', type=float, default=90)

    # Learning params
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--deterministic', type=int,  default=1)

    # Anything under torch.optim works. e.g., 'Adam' and 'SGD'
    parser.add_argument('--optimizer', type=str, default = 'Adam')
    parser.add_argument('--learning_rate', type=float, default = 0.002)
    parser.add_argument('--batch_size', type=int, default = 128)
    # Anything under torch.nn will work see https://pytorch.org/docs/stable/nn.html#loss-functions
    parser.add_argument('--loss_func', type=str, default = 'CrossEntropyLoss')

    # Algortihm params
    parser.add_argument('--recovery_perc', type=float, default = 0.75)
    parser.add_argument('--phase_epochs', type=int, default = 5)
    parser.add_argument('--reinit',  type=int, default = 1)
    parser.add_argument('--tau_schedule', type=str, default = 'cosine_anneling', choices=['cosine_anneling', 'exp_decay','linear'])
    parser.add_argument('--tau_param',  type=float, default = 30)
    parser.add_argument('--grow_init', type=str, default = 'zero_init', choices=['zero_init'])
    parser.add_argument('--grow_method', type=str, default = 'random', choices=['random']) 

    # Output head params
    parser.add_argument('--multihead', type=int, default = 1)
    parser.add_argument('--output_mechanism', type=str, default = 'vanilla')

    # Log params
    parser.add_argument('--log_prefix', type=str, default = 'Mlp')
    parser.add_argument('--log_suffix', type=str, default = 'verbose6')
    # 0 = No log
    # 1 = Accuracies, #stable/#plastix and model checkpoing after learning all tasks
    # 2 = "1" and Accuracies on all earlier tasks and #stable/#plastic after each task
    # 3 = "2" and model checkpoint and average_activations after each task
    # 4 = "3" and #stable, #plastic, #candidate stable units after each phase
    # 5 = "4" and accuracies on current task for each phase
    # 6 = "5" and model checkpoints and average_activations
    parser.add_argument('--verbose_logging', type=int, default = '6', choices=[0, 1, 2, 3, 4, 5, 6])

    return parser.parse_args()

def get_model_config_dict(args: argparse.Namespace) -> dict:
    return getattr(model_config, args.model)


def get_log_param_dict(args: argparse.Namespace) -> dict:
    return {
        "LogPath":  config.LOG_PATH,
        "DirName": args.log_prefix + "_" + args.experiment_name + "_" + args.log_suffix,
        "save_activations_task": args.verbose_logging in [3, 4, 5, 6],
        "save_activations_phases": args.verbose_logging in [6],
        "save_model_phase": args.verbose_logging in [6],
        "eval_model_phase": args.verbose_logging in [5, 6],
        "save_model_task": args.verbose_logging in [3, 4, 5, 6],
        "write_phase_log": args.verbose_logging in [4, 5, 6],
        "write_task_log": args.verbose_logging in [2, 3, 4, 5, 6],
        "write_sequence_log": args.verbose_logging in [1, 2, 3, 4, 5, 6],
        "no_log": args.verbose_logging == 0
    }

def create_log_dirs(args: argparse.Namespace, log_params: dict) -> None:
    dirpath = os.path.join(log_params["LogPath"], log_params["DirName"])
    # Remove existing files/dirs
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    
    if log_params["no_log"]:
        return
    # Create log dirs and save experiment args
    os.makedirs(dirpath)
    with open(os.path.join(dirpath, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    if log_params["write_task_log"]:
        for task_id in range(1, args.number_of_tasks +1):
            os.makedirs(os.path.join(dirpath, "Task_{}".format(task_id)))


def get_experience_streams(args: argparse.Namespace) -> Tuple[GenericCLScenario, int, int]:
    if args.dataset == "SplitMNIST":
        stream = SplitMNIST(n_experiences = args.number_of_tasks, seed = args.seed, dataset_root=config.DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        return (stream_with_val, 784, 10)
    if args.dataset == "SplitFMNIST":
        stream = SplitFMNIST(n_experiences = args.number_of_tasks, seed = args.seed, dataset_root=config.DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        return (stream_with_val, 784, 10)
    if args.dataset == "SplitCIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        stream = SplitCIFAR10(n_experiences = args.number_of_tasks, seed = args.seed, dataset_root=config.DATASET_PATH, fixed_class_order=list(range(10)),
                              train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        return (stream_with_val, 3, 10)
    if args.dataset == "SplitCIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        stream = SplitCIFAR100(n_experiences = args.number_of_tasks, seed = args.seed, dataset_root=config.DATASET_PATH, fixed_class_order=list(range(100)),
                               train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        return (stream_with_val, 3, 100)
    raise Exception("Dataset {} is not defined!".format(args.dataset))
