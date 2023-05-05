# adding auto encoder class to train and get a dense embedding for clustering
# and figure out topic words and do theme classification
import torch

INPUT_SHAPE = 768 # get the combined vector shape


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SHAPE, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),

        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, INPUT_SHAPE),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

