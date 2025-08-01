# FEDMED/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=1, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        layers = []
        current_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_model_template(input_dim):
    from config_tenseal import HIDDEN_SIZES, DROPOUT_RATE # <--- Changed
    return MLP(input_dim, HIDDEN_SIZES, output_dim=1, dropout_rate=DROPOUT_RATE)