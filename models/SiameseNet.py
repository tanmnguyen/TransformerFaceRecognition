import torch 
import torch.nn as nn 
class SiameseNet(nn.Module):
    def __init__(self, encoder: nn.Module, loss: nn.Module):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
        self.loss = loss

    def forward(self, anchor, positive, negative):
        anchor   = self.encoder(anchor)
        positive = self.encoder(positive)
        negative = self.encoder(negative)
        loss = self.loss(anchor, positive, negative)

        return loss 
    
    def get_param_groups(self, default_lr):
        try:
            return self.encoder.get_param_groups()
        except:
            return [{"params": self.encoder.parameters(), "lr": default_lr}]