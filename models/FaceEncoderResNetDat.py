import torch 
import einops 
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .DAT.nat import NeighborhoodAttention2D
from .DAT.dat_blocks import DAttentionBaseline 

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    
class NDat(nn.Module):
    def __init__(self, in_channel, h, w) -> None:
        super().__init__()

        self.nat = NeighborhoodAttention2D(dim=in_channel, kernel_size=3, num_heads=4, attn_drop=0., proj_drop=0.)
        self.dat = DAttentionBaseline(
            q_size=(h,w), kv_size=h, n_heads=4, n_head_channels=in_channel // 4, n_groups=2,
            attn_drop=0., proj_drop=0., stride=1, 
            offset_range_factor=3, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=5, log_cpb=False
        )
        self.relu = nn.ReLU()
        self.down_proj = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, 3, 2, 1, bias=False),
            LayerNormProxy(in_channel * 2)
        )

    def forward(self, x):
        x_transformer, _, _ = self.nat(x)
        x_transformer, _, _ = self.dat(x_transformer)
        x_transformer = self.relu(x + x_transformer)
        x_transformer = self.down_proj(x_transformer)

        return x_transformer

class FaceEncoderResnetDat(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.4):
        super().__init__()
        # CNN backbone 
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1 
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1

        self.layer2 = resnet.layer2
        self.ndat2 = NDat(64, 28, 28)

        self.layer3 = resnet.layer3
        self.ndat3 = NDat(128, 14, 14)

        self.layer4 = resnet.layer4
        self.ndat4 = NDat(256, 7, 7)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7 , hidden_dim)


    def forward(self, x):
        # extract features using CNN backbone 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        y = self.layer3(x)
        z = self.ndat3(x)
        x = y + nn.ReLU()(z)

        y = self.layer4(x)
        z = self.ndat4(x)
        x = y + nn.ReLU()(z)

        # map to latent space
        x = self.dropout(x)
        out = torch.flatten(x, 1)
        out = self.fc(out)
        
        return out
    

    def get_fine_tuned_param_groups(self):
        # only return fc layer for fine tuning
        return [
            {"params": self.fc.parameters(), "lr": 1e-4}
        ]
    
    def get_param_groups(self):
        resnet_lr = 1e-4 
        ndat_lr = 1e-3
        return [
            {"params": self.conv1.parameters(), "lr": resnet_lr},
            {"params": self.bn1.parameters(), "lr": resnet_lr},
            {"params": self.layer1.parameters(), "lr": resnet_lr},
            {"params": self.layer2.parameters(), "lr": resnet_lr},
            {"params": self.ndat2.parameters(), "lr": ndat_lr},
            {"params": self.layer3.parameters(), "lr": resnet_lr},
            {"params": self.ndat3.parameters(), "lr": ndat_lr},
            {"params": self.layer4.parameters(), "lr": resnet_lr},
            {"params": self.ndat4.parameters(), "lr": ndat_lr},
            {"params": self.fc.parameters(), "lr": ndat_lr},
        ]