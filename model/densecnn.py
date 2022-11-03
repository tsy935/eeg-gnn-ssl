import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.dense_inception.dense_inception import *


class DenseCNN(nn.Module):
    def __init__(self, params, data_shape, num_classes):
        super(DenseCNN, self).__init__()
        self.type = params.type
        self.data_shape = data_shape
            
        if (self.type == "dense_inception"):
            self.dense_inception = DenseInception(params, data_shape, num_classes=num_classes)
        else:
            raise NotImplementedError
        
    def forward(self, s):

        if (self.type == "dense_inception"):
            return self.dense_inception(s)  
        else:
            raise NotImplementedError