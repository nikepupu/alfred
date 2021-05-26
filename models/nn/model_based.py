import numpy as np

import torch
from torch import nn
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class LookAheadPolicy(nn.Module):
    def __init__(self, input_dimm, hidden_dim, output_dim):
        super(LookAheadPolicy, self).__init__()
        self.input_dimm = input_dimm
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dimm, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dimm)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return x
    


    



    
