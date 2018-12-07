import torch
from torch import nn

class ResidualGRU(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ResidualGRU, self).__init__()
        self.rnn = nn.GRU(*args, **kwargs)
        self.mapping = nn.Linear(self.rnn.hidden_size,
                self.rnn.input_size)

    def forward(self, x, h0=None):
        y, h = self.rnn(x, hx=h0)
        y = self.mapping(y)
        return y + x, h

    def forward_all(self, x, h0=None):
        y, h = self.rnn(x, hx=h0)
        y = self.mapping(y)
        return y + x, h, y


class ResidualModel(nn.Module):

    def __init__(self, input_size, num_blocks, hidden_size, num_layers_per_block):
        super(ResidualModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = [ResidualGRU(hidden_size, hidden_size, num_layers_per_block) for n in range(num_blocks)]
        for n, block in enumerate(self.blocks):
            self.add_module('Block{}'.format(n), block)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h, _ = block(h)
        return self.output_layer(h)

    def forward_iterative(self, x):
        h = self.input_layer(x)
        outputs = []
        for block in self.blocks:
            h, _ = block(h)
            outputs.append(self.output_layer(h))
        return outputs

    def forward_all(self, x):
        h = self.input_layer(x)
        outputs = []
        residuals = []
        for block in self.blocks:
            h, state, residual = block.forward_all(h)
            outputs.append(self.output_layer(h))
            residuals.append(residual)
        return outputs, residuals


if __name__ == '__main__':
    model = ResidualModel(20, 3, 10, 1)

