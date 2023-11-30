import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(__class__, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.rnn = nn.RNN(
            n_input,
            n_hidden,
            n_layers,
            batch_first=True,
            nonlinearity='relu'
        )
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, src, initial_hidden):
        output, hidden = self.rnn(src.unsqueeze(1), initial_hidden.detach())

        slice = output[:, -1, :]
        output = self.output_layer(slice)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_()
