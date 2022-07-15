# Import libraries
import unittest
import torch
from torch import nn
from torchmodel import Model
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

# Set the reproducibility
torch.manual_seed(42)

# Initialize train and validation data
train = mnist.MNIST(root='testDataset', train=True, download=True, transform=ToTensor())
valid = mnist.MNIST(root='testDataset', train=False, download=True, transform=ToTensor())


class CreateDatast(Dataset):
    def __init__(self, nrow, ncol, transforms = None, train: bool = True) -> None:
        super(CreateDatast, self).__init__()
        self.x = torch.randn(nrow, ncol)
        self.y = 1 + 2 * 1.0 * torch.randn(nrow, 1) * torch.rand(1)
        self.len = nrow
        self.transforms = transforms
        self.train = train
    
    def __len__(self) -> int:
        return self.len
        
    def __getitem__(self, index: int):
        if self.train:
            if self.transforms:
                transformed_x = self.transforms(self.x[index,  :])
                return transformed_x, self.y[index]
            return self.x[index, :], self.y[index]
            
        else:
            if self.transforms:
                transformed_x = self.transforms(self.x[index, :])
                return transformed_x
            else:
                return self.x[index, :]
                
                
# Construct train and validation dataloader
train_dataloader = DataLoader(train, shuffle=True)
valid_dataloader = DataLoader(valid, shuffle=True)

# Instantiate the Model object
model = Model([
    # Transpose Input data
    #nn.Flatten(),
    
    # Input layer
    nn.Linear(in_features=8, out_features=32),
    nn.ReLU(),  # Activation function
    nn.Dropout(.4),

    # First hidden layer
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),  # Activation function
    nn.Dropout(.4),  # Drop same pixel

    # Output layer
    nn.Linear(in_features=32, out_features=1)
])

model.compile(
    optimize=torch.optim.Adam(model.parameters()),
    loss=nn.MSELoss(),
    )

# Construct a dataset
data = CreateDatast(100, 8)
train_load = DataLoader(dataset=data, shuffle=True)

data = CreateDatast(100, 8, train=False)
test_load = DataLoader(dataset=data, shuffle=False)

model.fit(train_load, epochs=2)
#print(model.predict(test_load))


#class MyTestCase(unittest.TestCase):
    #def test_something(self):
        #self.assertEqual(True, False)  # add assertion here


#if __name__ == '__main__':
    #unittest.main()
