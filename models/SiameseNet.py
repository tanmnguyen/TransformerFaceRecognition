import torch 
import torch.nn as nn 
class SiameseNet(nn.Module):
    def __init__(self, encoder: nn.Module, loss: nn.Module):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
        self.loss = loss
        # self.branches = nn.Parallel(
        #     self.encoder, 
        #     self.encoder, 
        #     self.encoder
        # )

    def forward(self, anchor, positive, negative):
        anchor   = self.encoder(anchor)
        positive = self.encoder(positive)
        negative = self.encoder(negative)
        # anchor, positive, negative = self.branches(anchor, positive, negative)
        loss = self.loss(anchor, positive, negative)

        return loss 