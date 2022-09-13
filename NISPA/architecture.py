from argparse import Namespace
import torch.nn as nn
import torch
from typing import List, Tuple, Dict
from numpy import typing as np_type
import copy
import numpy as np

from NISPA.model_utils import MaskedConv2dDynamic, MaskedLinearDynamic, get_output_layer, get_device

class Network(nn.Module):
    def __init__(self, config_dict: Dict, input_size: int, output_size: int, args: Namespace) -> None:
        super(Network, self).__init__()
        self.config_dict = config_dict
        self.conv_layers = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()
        self.input_size = input_size

        # Continual Learning Attr
        self.classes_seen_so_far = []
        self.list_stable_units = []
        self.current_stable_and_candidate_units = []
        self.freeze_masks = []

        # Layers
        if "conv" in self.config_dict:
            prev_layer_out = input_size
            for ops, layer_config in zip(self.config_dict["conv"]["ops"], self.config_dict["conv"]["layers"]):
                out_channels, kernel_size = layer_config
                self.conv_layers.append(MaskedConv2dDynamic(prev_layer_out, out_channels, kernel_size))
                [self.conv_layers.append(op) for op in ops]
                prev_layer_out = out_channels
        if "hidden_linear" in self.config_dict:
            prev_layer_out = self.config_dict["conv"]["conv2lin_size"] if "conv" in self.config_dict else input_size 
            for ops, num_units in zip(self.config_dict["hidden_linear"]["ops"], self.config_dict["hidden_linear"]["layers"]):
                self.hidden_linear.append(MaskedLinearDynamic(prev_layer_out, num_units))
                [self.hidden_linear.append(op) for op in ops]
                prev_layer_out = num_units
        else:
            raise Exception("Network should have at least 1 hidden linear layer.")
        self.output_layer = get_output_layer(args, self.config_dict["hidden_linear"]["layers"][-1], output_size)
        self.output_size = output_size
        self._initialize_weights()



    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (MaskedLinearDynamic, MaskedConv2dDynamic)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_masks: List[torch.Tensor] , bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinearDynamic, MaskedConv2dDynamic)):
                m.set_mask(weight_masks[i],bias_masks[i])
                i = i + 1


    def add_new_stable_units(self, list_new_stable_units: List[List]) -> None:
        if len(self.list_stable_units) == 0:
            self.list_stable_units = list_new_stable_units
        else:
            self.list_stable_units = [list(set(new_stable).union(old_stable))  for new_stable, old_stable in zip(list_new_stable_units, self.list_stable_units)]

    def add_seen_classes(self, classes: List) -> None:
        new_classes = set(self.classes_seen_so_far)
        for cls in  classes:
            new_classes.add(cls)
        self.classes_seen_so_far = list(new_classes)

    def forward(self, x: torch.Tensor, current_classes = None) -> torch.Tensor:
        # Feedforward conv
        if "conv" in self.config_dict:
            for layer in self.conv_layers: x = layer(x)
            x = x.view(-1, self.config_dict["conv"]["conv2lin_size"])
        else:
            x = x.view(-1, self.input_size) # Flatten if MLP
        # Feedforward hidden linear
        for layer in self.hidden_linear: x = layer(x)
        # Feedforward output layer
        x = self.output_layer(x)
        # Mask non-task output units
        if current_classes:
            mask = torch.zeros(self.output_size)
            mask[current_classes] = 1
            x = x * mask.to(get_device())
        return x

    def forward_activations(self, x: torch.Tensor) ->  list[torch.Tensor]:
        activations = []
        # Feedforward conv
        if "conv" in self.config_dict:
            for layer in self.conv_layers:
                x = layer(x)
                if isinstance(layer, torch.nn.ReLU):
                    activations.append(copy.deepcopy(x.detach()))
                # Remove last relu and use Maxpool output instead
                # This makes Conv2Lin connectivity easy to handle
                if isinstance(layer, torch.nn.MaxPool2d):
                    activations.pop()
                    activations.append(copy.deepcopy(x.detach()))
            x = x.view(-1, self.config_dict["conv"]["conv2lin_size"])
        else:
            x = x.view(-1, self.input_size) # Flatten if MLP
        # Feedforward hidden linear
        for layer in self.hidden_linear:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU): activations.append(copy.deepcopy(x.detach()))

        # Feedforward output layer
        x = self.output_layer(x)
        activations.append(copy.deepcopy(x.detach()))
        return activations

    def get_weight_bias_masks_numpy(self) -> List[Tuple[np_type.NDArray[np.double], np_type.NDArray[np.double]]]:
        weights = []
        for module in self.modules():
            if isinstance(module,MaskedLinearDynamic) or isinstance(module, MaskedConv2dDynamic):
                weight_mask, bias_mask = module.get_mask()  # type: ignore
                weights.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy())) # type: ignore
        return weights

    def freeze_stable_to_stable(self) -> None:
        weights = self.get_weight_bias_masks_numpy()
        freeze_masks = []
        for i, (source_stable, target_stable) in enumerate(zip(self.list_stable_units[:-1], self.list_stable_units[1:])):
            source_stable, target_stable = np.array(source_stable, dtype=np.int32), np.array(target_stable, dtype=np.int32)
            mask_w = np.zeros(weights[i][0].shape)
            #Conv2Conv
            if len(weights[i][0].shape) == 4:
                for src_unit_stable in source_stable:
                    mask_w[target_stable, src_unit_stable, :, :] = 1
            #Conv2Linear or Linear2Linear
            else:
                 #Conv2Linear
                if len(weights[i-1][0].shape) == 4:
                    for src_unit_stable in source_stable:
                        conv2lin_kernel_size = self.config_dict["conv"]["conv2lin_mapping_size"]
                        mask_w[target_stable, src_unit_stable*conv2lin_kernel_size:(src_unit_stable + 1)*conv2lin_kernel_size] = 1
                else:
                    for src_unit_stable in source_stable:
                        mask_w[target_stable, src_unit_stable] = 1 
            mask_b = np.zeros(weights[i][1].shape)
            mask_b[target_stable] = 1
            freeze_masks.append((mask_w * weights[i][0], mask_b))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        freeze_masks = [(torch.tensor(w).to(torch.bool).to(device) , torch.tensor(b).to(torch.bool).to(device)) for w, b in freeze_masks]
        self.freeze_masks = freeze_masks

    def re_initialize_not_frozen(self):
        i = 0
        for m in self.modules():
            if isinstance(m, MaskedLinearDynamic) or isinstance(m, MaskedConv2dDynamic):
                old_weights = m.weight.data.clone().detach()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data[self.freeze_masks[i][0].clone().detach().to(torch.bool)] = old_weights[self.freeze_masks[i][0].clone().detach().to(torch.bool)]
                
                old_bias = m.bias.clone().detach()   # type: ignore
                nn.init.constant_(m.bias.data, 0)   # type: ignore
                m.bias.data[self.freeze_masks[i][1].clone().detach().to(torch.bool)] =  old_bias[self.freeze_masks[i][1].clone().detach().to(torch.bool)]   # type: ignore
                i += 1

    def compute_weight_sparsity(self):
        parameters = 0
        ones = 0
        for module in self.modules():
            if isinstance(module,MaskedLinearDynamic) or isinstance(module, MaskedConv2dDynamic):
                shape = module.weight.data.shape
                parameters += torch.prod(torch.tensor(shape))
                w_mask, _ = copy.deepcopy(module.get_mask())
                ones += torch.count_nonzero(w_mask)
        return float((parameters - ones) / parameters) * 100



def random_prune(network: Network, pruning_perc: float, skip_first_conv = True) -> Network:
    network = copy.deepcopy(network)
    pruning_perc = pruning_perc / 100.0
    weight_masks = []
    bias_masks = []
    first_conv_flag = skip_first_conv
    for module in network.modules():
        if isinstance(module, MaskedLinearDynamic):
            weight_masks.append(torch.from_numpy(np.random.choice([0, 1], module.weight.shape,
                                                                  p =  [pruning_perc, 1 - pruning_perc])))
            # We do not prune biases
            bias_masks.append(torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])))
        #Channel wise pruning Conv Layer
        elif isinstance(module, MaskedConv2dDynamic):
           connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                (module.weight.shape[0],  module.weight.shape[1]),
                                                p =  [0, 1] if first_conv_flag else [pruning_perc, 1 - pruning_perc]))
           first_conv_flag = False
           in_range, out_range = range(module.weight.shape[1]), range(module.weight.shape[0])
           kernel_shape = (module.weight.shape[2], module.weight.shape[3])
           filter_masks = [[np.ones(kernel_shape) if connectivity_mask[out_index, in_index] else np.zeros(kernel_shape)
                            for in_index in in_range]
                            for out_index in out_range]
           weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32))
           
           #do not prune biases
           bias_mask = torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])).to(torch.float32)  # type: ignore
           bias_masks.append(bias_mask)
    network.set_masks(weight_masks, bias_masks)
    network.to(get_device())
    return network