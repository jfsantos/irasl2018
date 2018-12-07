import torch
from torch import nn

class BaselineModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(BaselineModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h1 = self.input_layer(x)
        h2, _ = self.rnn(h1)
        return self.output_layer(h2)

