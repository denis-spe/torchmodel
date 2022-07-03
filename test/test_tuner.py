# import library
import torch
from torch import nn
from torchmodel_PyDen.torchmodel import Model
from tuner import Hyperparameter, Tuner

hyper = Hyperparameter()

def model():
    infeature = hyper.Choice('in_feature', [32, 64, 128, 256, 512])
    outfeature = hyper.Int('out_feature', 32, 512, 32)
    
    
    
    _model = Model(layers=[
        nn.Linear(in_features=infeature,    out_features=outfeature),
        nn.Linear(in_features=infeature, out_features=outfeature)
    ])
    
    lr = hyper.Float('lr', 0, 5)
    _model.compile(optimize=torch.optim.Adam(_model.parameters(), lr=lr),
    loss=torch.nn.CrossEntropyLoss()
    )
    return _model
    
    
    
tune = Tuner(
    model=model,
    hyperparameter=hyper,
    max_epoch=20
)

tune.study()

print(hyper.select)