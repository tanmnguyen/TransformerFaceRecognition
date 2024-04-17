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
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.nat1 = NeighborhoodAttention2D(dim=128, kernel_size=3, num_heads=4, attn_drop=0., proj_drop=0.)
        self.dat1 = DAttentionBaseline(
            q_size=(14,14), kv_size=14, n_heads=4, n_head_channels=128 // 4, n_groups=2,
            attn_drop=0., proj_drop=0., stride=1, 
            offset_range_factor=3, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=5, log_cpb=False
        )
        
        self.down_proj1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            LayerNormProxy(256)
        )

        self.nat2 = NeighborhoodAttention2D(dim=256, kernel_size=3, num_heads=4, attn_drop=0., proj_drop=0.)
        self.dat2 = DAttentionBaseline(
            q_size=(14,14), kv_size=14, n_heads=4, n_head_channels=256 // 4, n_groups=2,
            attn_drop=0., proj_drop=0., stride=1, 
            offset_range_factor=3, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=5, log_cpb=False
        )

        self.down_proj2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            LayerNormProxy(512)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7 , hidden_dim)


    def forward(self, x):
        # # extract features using CNN backbone 
        # x0 = self.cnn(x)['features']

        # # transform features using NAT and DAT blocks
        # x1_transformer, _, _ = self.nat1(x0) 
        # x1_transformer, _, _ = self.dat1(x1_transformer)
        # x1_transformer = self.relu(x0 + x1_transformer)
        # x1_transformer = self.down_proj1(x1_transformer)

        # # trasnform features using ResNet layer 3
        # x1_resnet = self.resnetLayer3(x0)
        # x1 = self.relu(x1_transformer + x1_resnet)

        # # transform features using NAT and DAT blocks
        # x2_transformer, _, _ = self.nat2(x1)
        # x2_transformer, _, _ = self.dat2(x2_transformer)
        # x2_transformer = self.relu(x1 + x2_transformer)
        # x2_transformer = self.down_proj2(x2_transformer)

        # # trasnform features using ResNet layer 4
        # x2_resnet = self.resnetLayer4(x1)
        # x2 = self.relu(x2_transformer + x2_resnet)

        # # flatten features
        # out = torch.flatten(x2, 1)
        # out = self.dropout(out)

        # # map to latent space
        # out = self.fc(out)

        # cnn backbone 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # map to latent space
        x = self.dropout(x)
        out = torch.flatten(x, 1)
        out = self.fc(out)
        
        return out