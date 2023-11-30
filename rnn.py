import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models.rnn2 import RNN


class ModelDataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.n_src = 3
        self.n_tgt = 1
        self.n_snapshot = self.n_src + self.n_tgt

    def __getitem__(self, item):
        snapshot = self.data[item:item + self.n_snapshot]
        src_data = snapshot[:self.n_src]
        tgt_data = snapshot[self.n_src:]
        return src_data, tgt_data

    def __len__(self):
        return len(self.data) - self.n_snapshot + 1


def main():
    from data import data
    dataset = ModelDataset(data)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = RNN(3, 64, 1, 1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    current_loose = 0
    loss_history = []

    for epoch in range(1, n_epochs + 1):

        for src, tgt in iter(loader):
            optimizer.zero_grad()

            hidden = model.init_hidden(1)
            output, hidden = model(src, hidden)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            current_loose += (loss.item() / 100)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{n_epochs}, Loss: {current_loose:.5f}")
            loss_history.append(current_loose)
            current_loose = 0

    _input_data = [8, 2, 9]
    _input = torch.tensor([_input_data], dtype=torch.float32)
    output, _ = model(_input, model.init_hidden(1))

    print(f"Input: {_input_data}, Predict: {round(output.item())}")


if __name__ == '__main__':
    main()









