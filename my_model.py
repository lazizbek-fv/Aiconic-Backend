# my_model.py
import torch

class YourModel(torch.nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.layer = torch.nn.Linear(100, 10)  # Replace this with your actual model architecture

    def forward(self, x):
        return self.layer(x)