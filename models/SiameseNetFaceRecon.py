import torch.nn as nn 
class SiameseNetFaceRecon(nn.Module):
    def __init__(self, encoder: nn.Module, loss: nn.Module, face_recon: nn.Module):
        super(SiameseNetFaceRecon, self).__init__()
        self.loss = loss
        self.encoder = encoder
        self.face_recon = face_recon

    def forward(self, anchor, positive, negative):
        anchor_latent   = self.encoder(anchor)
        positive_latent = self.encoder(positive)
        negative_latent = self.encoder(negative)
        loss, dis_pos, dis_neg = self.loss(anchor_latent, positive_latent, negative_latent)

        recon = self.face_recon(anchor_latent)
        recon_loss = self.face_recon.loss(recon, anchor)

        loss = loss + recon_loss * 0.1
        return loss, dis_pos, dis_neg
    
    def get_param_groups(self, default_lr):
        try:
            return self.encoder.get_fine_tuned_param_groups()
        except:
            return [{"params": self.encoder.parameters(), "lr": default_lr}]