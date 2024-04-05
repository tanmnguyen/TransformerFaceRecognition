import torch.nn as nn 

from settings import settings 

class ClassifierNet(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss().to(settings.device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def loss(self, output, target):
        return self.loss_fn(output, target)
    