# impory the necessary library
import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class Accuracy:
    name  = "accuracy"
    
    def __call__(self, yhat: any, y: any):
        _, predict = torch.max(yhat, 1)
        return (predict.argmax(1) == y).sum().item()


@dataclass        
class RMSE:
        name = "rmse"
        
        def __call__(self, yhat, y):
            return torch.mean(yhat - y) ** 2 / len(y)

@dataclass        
class MAE:
        name = "mae"
        
        def __call__(self, yhat, y):
            return torch.mean(torch.abs(yhat - y))
        
        
 
