from argparse import Namespace
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader

from  NISPA.architecture import Network, get_device


def reset_frozen_gradients(network: Network) -> Network:
    mask_index = 0
    for module in network.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[network.freeze_masks[mask_index][0]] = 0  # type: ignore
            module.bias.grad[network.freeze_masks[mask_index][1]] = 0    # type: ignore
            mask_index = mask_index + 1
    return network


def test(network: Network, data_loader: DataLoader, classes_in_this_experience: List, report = False) -> float:
    network.eval()
    correct1 = 0
    with torch.no_grad():
        for data, target, _ in data_loader:
            data = data.to(get_device())
            target = target.to(get_device())
            output = network(data, classes_in_this_experience)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
    acc1 = 100.0 * correct1 / len(data_loader.dataset)  # type: ignore
    if report:
          print('Top 1 Accuracy = ', acc1)
    return acc1

def task_training(network: Network, loss: nn.Module, optimizer: ..., train_loader: DataLoader, args: Namespace, classes_in_this_experience: List) -> Network:

    for _ in range(args.phase_epochs):
        network.train()
        for data, target, _ in train_loader:
            data = data.to(get_device())
            target = target.to(get_device())
            optimizer.zero_grad()
            output = network(data, classes_in_this_experience)
            batch_loss = loss(output, target.long())
            batch_loss.backward()
            if network.freeze_masks:
                network = reset_frozen_gradients(network)
            optimizer.step()

    return network