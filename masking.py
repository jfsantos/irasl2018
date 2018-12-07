import torch
from torch import nn

class MaskingGRU(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MaskingGRU, self).__init__()
        self.rnn = nn.GRU(*args, **kwargs)
        self.gate = nn.Linear(args[0], args[0], bias=True)

    def forward(self, x, h0=None):
        y, h = self.rnn(x, hx=h0)
        return torch.sigmoid(self.gate(y)) * x, h

    def forward_all(self, x, h0=None):
        y, h = self.rnn(x, hx=h0)
        g = torch.sigmoid(self.gate(y))
        return g * x, h, g


class MaskingModel(nn.Module):

    def __init__(self, input_size, num_blocks, hidden_size, num_layers_per_block):
        super(MaskingModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = [MaskingGRU(hidden_size, hidden_size, num_layers_per_block) for n in range(num_blocks)]
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
        masks = []
        for block in self.blocks:
            h, state, mask = block.forward_all(h)
            outputs.append(self.output_layer(h))
            masks.append(mask)
        return outputs, masks


if __name__ == '__main__':
    model = MaskingModel(20, 3, 10, 1)
    x = torch.FloatTensor(3, 4, 20)
    y = model(x)

