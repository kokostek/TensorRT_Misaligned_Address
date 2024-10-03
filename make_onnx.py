import torch
import torch.onnx
from torch import nn


class SampleModel(nn.Module):

    def __init__(self, *, bias: bool, groups: int):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=256, out_channels=16,
            kernel_size=3, stride=1, padding=1,
            bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)


def main():

    configurations = [
        {'bias': False, 'groups': 1},
        {'bias': False, 'groups': 16},
        {'bias': True, 'groups': 1},
        {'bias': True, 'groups': 16},
    ]

    for args in configurations:

        model = SampleModel(**args)
        dummy_input = torch.randn(1, 256, 4, 4)
        onnx_file = f'bias={args["bias"]}_groups={args["groups"]}.onnx'

        torch.onnx.export(
            model, dummy_input, onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )


if __name__ == '__main__':
    main() 
