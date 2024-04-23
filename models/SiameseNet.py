import torch 
import torch.nn as nn 
class SiameseNet(nn.Module):
    def __init__(self, encoder: nn.Module, loss: nn.Module, face_reconstruction: nn.Module):
        super(SiameseNet, self).__init__()
        self.loss = loss
        self.encoder = encoder
        self.face_reconstruction = face_reconstruction 

    def forward(self, anchor, positive, negative, beta = 0.2):
        anchor_latent   = self.encoder(anchor)
        positive_latent = self.encoder(positive)
        negative_latent = self.encoder(negative)
        triplet_loss, dis_pos, neg_pos = self.loss(anchor_latent, positive_latent, negative_latent)

        # compute reconstruction loss 
        b = anchor.size(0)
        recon_face = self.face_reconstruction(anchor_latent)
        recon_loss = nn.MSELoss()(recon_face.contiguous().view(b, -1), anchor.contiguous().view(b, -1))
        
        # combine losses
        loss = triplet_loss + recon_loss * beta 

        return loss, dis_pos, neg_pos, triplet_loss, recon_loss
    
    def get_param_groups(self, default_lr):
        try:
            return self.encoder.get_param_groups()
        except:
            return [{"params": self.encoder.parameters(), "lr": default_lr}]