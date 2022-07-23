# import libraries
import time
import random
import progress # progress2
import torch, os, metrics
from typing import Iterator, Tuple, Dict, List, Callable
from torchinfo import summary
from torch.utils.data import DataLoader
from torch import nn
from progress.bar import IncrementalBar
from progress.colors import bold

def sleep():
    t = 0.01
    t += t * random.uniform(-0.1, 0.1)  # Add some variance
    time.sleep(t)

class Model(nn.Module):
    def __init__(self, layers: Iterator) -> None:
        self.__layers = layers
        super(Model, self).__init__()
        self.__stacked_layers = nn.Sequential(*self.__layers)
        self.model_training = True

    def forward(self, X):
        return self.__stacked_layers(X)

    def compile(self, optimize: any, loss: any, metrics, device: str = 'cpu') -> None:
        self.device = device
        self.optim = optimize
        self.loss = loss
        self.model = Model(self.__layers).to(self.device)
        self.metrics = metrics

    def summaries(self, input_size=None):
        if input_size:
            return summary(self, input_size=input_size)
        else:
            return summary(self)

    def train_process(self, data_loader, metric: Callable, verbose: bool = False) -> Tuple[float, float, float, int]:
        """
        Train model on train_data
        """
        # Indicating the model to training
        # self.model.train()

        # Number of images in  data_loader: size
        size = len(data_loader.dataset)

        # Initialize the metric variable
        total_score, total_loss, count_label, current = 0, 0, 0, 0
        
        start = time.time()
        # iterate over the data_loader
        for batch, (X, y) in enumerate(data_loader):
            # Switch to device
            X, y = X.to(self.device), y.to(self.device)

            # Make prediction
            yhat = self.model(X)

            # *** Backpropagation Process ***

            # Compute error by measure the degree of dissimilarity
            # from obtained result in target
            criterion = self.loss(yhat, y)

            # Reset the gradient of the model parameters
            # Gradients by default add up; to prevent double-counting,
            # we explicitly zero them at each iteration.
            self.optim.zero_grad()

            # Back propagate the prediction loss to deposit the gradient of loss
            # for learnable parameters
            criterion.backward()

            # Adjust the parameters by gradient collected in the backward pass
            self.optim.step()

            # Count number of labels
            count_label += len(y)

            # sum each loss to total_loss variable
            total_loss += criterion.item()

            #_, predict = torch.max(yhat, 1)

            # Add every accuracy on total_acc
            #total_score += (predict == y).sum().item()
            total_score += metric(yhat, y)

            if batch % 100 == 0:
                current += (batch / size)

        stop = time.time()
        time_taken = round(stop - start, 3)

        return (total_loss / count_label,
                total_score / count_label, time_taken,
                int(round(current * 100))
                )

    def evaluate(self, data_loader) -> Tuple[float, float]:
        """
        Evaluation model with validation data
        """
        # Directing model to evaluation process
        self.model.eval()

        # Instantiate metric variables
        total_loss, total_acc, count_labels = 0, 0, 0

        # Disabling gradient calculation
        with torch.no_grad():
            for X, y in data_loader:
                # Set to device
                X, y = X.to(self.device), y.to(self.device)

                # Make prediction
                predictions = self.model(X)

                # Compute the loss(error)
                criterion = self.loss(predictions, y)

                # Add number of label to count_labels
                count_labels += len(y)

                # Add criterion loss to total_loss
                total_loss += criterion.item()

                # Sum accuracy to total_acc
                total_acc += (predictions.argmax(1) == y).sum().item()

            # Finally, return total_loss and total_acc which each is divided by
            # count_labels
            return total_loss / count_labels, total_acc / count_labels

    def fit(
            self,
            train_data: DataLoader,
            epochs: int = 1,
            validation_data: DataLoader = None,
            verbose: bool = True,
            callbacks: list = None
    ) -> Dict[str, list[float]]:
        """
        The Fit method make use of train data and
        validation data if provided
        :param callbacks: (List) Pass a callback in list or None
        :param verbose: (bool) Model training progress
        :param validation_data: (DataLoader) Data to validate the model
        :param epochs: (int) number of training iteration
        :param train_data: (DataLoader) Data to train the model
         preformed: model fitting
        """
        # Initializing variable for storing metric score
        metrics = {}
        score_list = []
        loss_list = []
        valid_acc_list = []
        valid_loss_list = []

        # loop through the epoch
        for epoch in range(epochs):
            if verbose:
                print(f"\033[1m\nEpoch {epoch + 1}/{epochs}\033[0m")
                bar = IncrementalBar(bold(""), color='white')
                for i in bar.iter(range(100)):
                    sleep()
            
            # Train the data                
            train = self.train_process(train_data, verbose=verbose, metric=self.metrics)

            # Instantiate train loss and accuracy
            train_loss = round(train[0], 6)
            train_score = round(train[1].item(), 5)
            
            if verbose:
                print(f" loss: {train_loss} - {self.metrics.name}: {train_score} ETA: {round(train[2] * 100, 2)}ms", end="")

            # Storing the model score
            score_list.append(train_score)
            loss_list.append(train_loss)

            if validation_data:
                valid = self.evaluate(validation_data)
                # Instantiate train loss and accuracy
                valid_loss = round(valid[0], 6)
                valid_acc = round(valid[1], 4)

                if verbose:
                    if self._is_continuous:
                        print(f"- val_loss: {valid_loss} ",
                          end="")
                    else:
                        print(f"- val_loss: {valid_loss} - val_acc: {valid_acc}")

                # Store the score
                valid_loss_list.append(valid_loss)
                valid_acc_list.append(valid_acc)

            if not self.model_training:
                # Break the training loop if model_training is false
                break

            if callbacks:
                for callback in callbacks:
                    callback(self, metrics)

        metrics["acc"] = score_list
        metrics["loss"] = loss_list

        if validation_data:
            metrics["val_acc"] = valid_acc_list
            metrics["val_loss"] = valid_loss_list

#         if not verbose:
#             print(f"\nEpoch {epoch + 1}/{epochs} ")
#             print(f"{type(self.loss).__name__} loss: {metrics['loss'][-1]} - acc: {metrics['acc'][-1]} ", end="")
#             print(f"val_loss: {metrics['val_loss'][-1]} - val_acc: \
# {metrics['val_acc'][-1]} ", end="")

        return metrics

    def predict(self, y: torch.tensor) -> torch.tensor:
        # list storage for predictions
        predictions = []

        # Indicate to evaluation process
        # self.model.eval()

        # Don't use the gradient
        with torch.no_grad():
            # Instantiate the dataset
            data = y.dataset
            
            # Loop over the values in y
            for val in data:
                # switch to device
                val = val.to(self.device)
                if self._is_continuous:
                    # Make prediction
                    pred = self.model(val)
                    
                    # Append the predictions to the list
                    predictions.append(pred)
                    
                else:
                    # Make prediction
                    probability = self.model(val)

                    # probability variable returns probability
                    # Therefor convert it to actual value
                    prediction = torch.argmax(probability, 1).item()

                    # Add prediction to predictions list
                    predictions.append(prediction)
        return predictions

    

























#                                   God Bless You