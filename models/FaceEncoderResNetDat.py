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
    def __init__(self):
        super().__init__()
        return_nodes = {
            "layer2.1.bn2": "features"
        }
        
        # CNN backbone 
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = create_feature_extractor(resnet, return_nodes=return_nodes)

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

        # self.down_proj2 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, 2, 1, bias=False),
        #     LayerNormProxy(512)
        # )

        self.relu = nn.ReLU()

        self.final_conv = resnet.layer4 


    def forward(self, x):
        # extract features using CNN backbone 
        x0 = self.cnn(x)['features']

        # transform features using DAT blocks
        x1, _, _ = self.nat1(x0) 
        x1, _, x = self.dat1(x1)
        x1 = self.relu(x0 + x1)
        x1 = self.down_proj1(x1)

        x2, _, _ = self.nat2(x1)
        x2, _, x = self.dat2(x2)
        x2 = self.relu(x1 + x2)
        # x2 = self.down_proj2(x2)

        x3 = self.final_conv(x2)
        x3 = torch.flatten(x3, 1)
        
        return x3 