from torch import nn

Mlp_Vanilla = {
    "hidden_linear": {
        "ops": [(nn.ReLU(), ), (nn.ReLU(), ), (nn.ReLU(), )],
        "layers": [400, 400, 400]
    },
}

Conv_Vanilla = {
    "conv": {
        "ops": [(nn.ReLU(), ),  (nn.ReLU(), nn.MaxPool2d(2)), (nn.ReLU(), ), (nn.ReLU(), nn.MaxPool2d(2))],
        # (out_channels, kernel_size)
        "layers": [(64, 3), (64, 3), (128, 3), (128, 3)],
        # Number of flat features after last conv layer.
        "conv2lin_size": 128 * 64,
        "conv2lin_mapping_size": 64
    },

    "hidden_linear": {
        "ops": [(nn.ReLU(), )],
        "layers": [1024]
    },
}