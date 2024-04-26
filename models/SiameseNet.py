import torch 
import torch.nn as nn 
class SiameseNet(nn.Module):
    def __init__(self, encoder: nn.Module, loss: nn.Module):
        super(SiameseNet, self).__init__()
        self.loss = loss
        self.encoder = encoder

    def forward(self, anchor, positive, negative):
        anchor_latent   = self.encoder(anchor)
        positive_latent = self.encoder(positive)
        negative_latent = self.encoder(negative)
        loss, dis_pos, dis_neg = self.loss(anchor_latent, positive_latent, negative_latent)
        return loss, dis_pos, dis_neg, torch.tensor(0.0)
    
    def get_param_groups(self, default_lr):
        try:
            return self.encoder.get_param_groups()
        except:
            return [{"params": self.encoder.parameters(), "lr": default_lr}]