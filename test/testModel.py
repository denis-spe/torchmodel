# Import libraries
import torch
from torch import nn
from torchmodel import Model
from metrics import MAE
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

# Set the reproducibility
torch.manual_seed(42)

# Initialize train and validation data
train = mnist.MNIST(root='testDataset', train=True, download=True, transform=ToTensor())

valid = mnist.MNIST(root='testDataset', train=False, download=True, transform=ToTensor())

# Couple the dataset into batch
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)

valid_dataloader = DataLoader(train, batch_size=64, shuffle=False)

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
    nn.Linear(in_features=256, out_features=10)
])

# Compile the model
#model.compile(
#    optimize=torch.optim.Adam(model.parameters()),
#    loss=nn.CrossEntropyLoss(),
#    )

# Fit the mnist data  
#model.fit(train_dataloader)




# Test 2 #####################

# Create dataset
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
    metrics=MAE(),
    device=None
    )

# Construct a dataset
data = CreateDatast(100, 8)
train_load = DataLoader(dataset=data, shuffle=True)

data = CreateDatast(100, 8, train=False)
test_load = DataLoader(dataset=data, shuffle=False)




# Fit the data
model.fit(train_load, epochs=10, verbose=True)
# print(model.predict(test_load))

