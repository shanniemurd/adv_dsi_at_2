#import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))


    
    
class PytorchMultiClass(nn.Module):
    def __init__(self, num_feature, num_class):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_out = nn.Linear(128, num_class)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
    
    

#class PytorchMultiClass(nn.Module):
#    def __init__(self, num_feature, num_class):
#        super(PytorchMultiClass, self).__init__()
#        
#        self.layer_1 = nn.Linear(num_feature, 512)
#        self.layer_2 = nn.Linear(512, 128)
#        self.layer_out = nn.Linear(128, num_class)
#        self.softmax = nn.Softmax(dim=1)
#
#    def forward(self, x):
#        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
#        x = self.layer_out(x)
#        return self.softmax(x)#

    
def train_classification(train_data, model, criterion, optimizer, batch_size, device, scheduler=None, generate_batch=None):
    """Train a Pytorch multi-class classification model

    Parameters
    ----------
    train_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    optimizer: torch.optim
        Optimizer
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    scheduler : torch.optim.lr_scheduler
        Pytorch Scheduler used for updating learning rate
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """
    
    # Set model to training mode
    model.train()
    train_loss = 0
    train_acc = 0
    
    # Create data loader
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:

        # Reset gradients
        optimizer.zero_grad()
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Make predictions
        output = model(feature)
        
        # Calculate loss for given batch
        loss = criterion(output, target_class.long())

        # Calculate global loss
        train_loss += loss.item()
        
        # Calculate gradients
        loss.backward()

        # Update Weights
        optimizer.step()
        
        # Calculate global accuracy
        train_acc += (output.argmax(1) == target_class).sum().item()

    # Adjust the learning rate
    if scheduler:
        scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def test_classification(test_data, model, criterion, batch_size, device, generate_batch=None):
    """Calculate performance of a Pytorch multi-class classification model

    Parameters
    ----------
    test_data : torch.utils.data.Dataset
        Pytorch dataset
    model: torch.nn.Module
        Pytorch Model
    criterion: function
        Loss function
    bacth_size : int
        Number of observations per batch
    device : str
        Name of the device used for the model
    collate_fn : function
        Function defining required pre-processing steps

    Returns
    -------
    Float
        Loss score
    Float:
        Accuracy Score
    """    
    
    # Set model to evaluation mode
    model.eval()
    test_loss = 0
    test_acc = 0
    
    # Create data loader
    data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
    
    # Iterate through data by batch of observations
    for feature, target_class in data:
        
        # Load data to specified device
        feature, target_class = feature.to(device), target_class.to(device)
        
        # Set no update to gradients
        with torch.no_grad():
            
            # Make predictions
            output = model(feature)
            
            # Calculate loss for given batch
            loss = criterion(output, target_class.long())

            # Calculate global loss
            test_loss += loss.item()
            
            # Calculate global accuracy
            test_acc += (output.argmax(1) == target_class).sum().item()

    return test_loss / len(test_data), test_acc / len(test_data)


class PytorchCNN(nn.Module):
    def __init__(self):
        super(PytorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)