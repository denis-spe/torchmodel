# Import libraries
import unittest
import torch
from torch import nn
from src.torchmodel.torchmodel import Model
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Set the reproducibility
torch.manual_seed(42)

# Initialize train and validation data
train = mnist.MNIST(root='testDataset', train=True, download=True, transform=ToTensor())
valid = mnist.MNIST(root='testDataset', train=False, download=True, transform=ToTensor())

# Construct train and validation dataloader
train_dataloader = DataLoader(train, shuffle=True)
valid_dataloader = DataLoader(valid, shuffle=True)

# Instantiate the Model object
model = Model([
    # Transpose Input data
    nn.Flatten(),

    # Input layer
    nn.Linear(in_features=28 * 28, out_features=256),
    nn.ReLU(),  # Activation function
    nn.Dropout(.4),

    # First hidden layer
    nn.Linear(in_features=256, out_features=256),
    nn.ReLU(),  # Activation function
    nn.Dropout(.4),  # Drop same pixel

    # Output layer
    nn.Linear(in_features=256, out_features=10),
    nn.Softmax()
])

model.compile(
    optimize=torch.optim.Adam(model.parameters()),
    loss=nn.CrossEntropyLoss(),
    )

print(model.train_process(train_dataloader))


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
