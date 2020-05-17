# -*- coding: utf-8 -*-

#class that creates neural network using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,5,kernel_size=3) #first convolution layer
        self.conv2 = nn.Conv2d(5,10,kernel_size=3) #second convolution layer
        self.conv2_drop = nn.Dropout2d() #dropout layer
        self.fc1 = nn.Linear(40,20) #first fully-connected layer
        self.fc2 = nn.Linear(20,5) #second fully-connected layer
        self.lsm = nn.LogSoftmax(dim=1) #log sofmax for output
        
        
    def forward(self, x): #feet-forward
        x = func.relu(func.max_pool2d(self.conv1(x), 2)) #perform convolution and pool
        #x = func.relu(func.max_pool2d(self.conv2_drop(self.conv2(x)) , 2)) #convolve, dropout, and pool
        x = func.relu(func.max_pool2d(self.conv2(x) , 2)) #convolve and pool
        x = x.view(-1,40) #resize for linear layers
        x = func.relu(self.fc1(x)) #first linear layer
        x = self.fc2(x) #second linear layer
        return self.lsm(x) #return log softmax output

        
    def backprop(self, inputs , targets, loss, optimizer): #back propagation
        self.train()
        outputs = self(inputs)
        obj_val = loss(outputs , targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    
    def test(self, inputs, targets, loss): #test model on test data
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
            test_val = loss(outputs, targets)
        return test_val
    
    
    