import torch
import torch.nn as nn


class RNN(nn.Module):
    # you can also accept arguments in your model constructor
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(RNN, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.i2h = nn.Linear(n_input + n_hidden, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)

    def forward(self, src, last_hidden):
        _input = torch.cat((src, last_hidden), 1)
        hidden = self.i2h(_input)
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_()
