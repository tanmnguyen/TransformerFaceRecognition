import torch 
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from .DAT.nat import NeighborhoodAttention2D
from .DAT.dat_blocks import DAttentionBaseline 

class FaceEncoderResnetDat(nn.Module):
    def __init__(self):
        super().__init__()
        return_nodes = {
            "layer3.1.bn2": "features"
        }

        # CNN backbone 
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = create_feature_extractor(resnet, return_nodes=return_nodes)
        self.nat1 = NeighborhoodAttention2D(dim=256, kernel_size=3, num_heads=4, attn_drop=0., proj_drop=0.)
        self.dat1 = DAttentionBaseline(
            q_size=(14,14), kv_size=14, n_heads=4, n_head_channels=256 // 4, n_groups=2,
            attn_drop=0., proj_drop=0., stride=1, 
            offset_range_factor=3, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=5, log_cpb=False
        )
        self.nat2 = NeighborhoodAttention2D(dim=256, kernel_size=3, num_heads=4, attn_drop=0., proj_drop=0.)
        self.dat2 = DAttentionBaseline(
            q_size=(14,14), kv_size=14, n_heads=4, n_head_channels=256 // 4, n_groups=2,
            attn_drop=0., proj_drop=0., stride=1, 
            offset_range_factor=3, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=5, log_cpb=False
        )
        self.conv_down = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # extract features using CNN backbone 
        x0 = self.cnn(x)['features']

        # transform features using DAT blocks
        x, _, _ = self.nat1(x0) 
        # x, _, _ = self.dat1(x)
        # x, _, _ = self.nat2(x)
        # x, _, _ = self.dat2(x)

        # x = self.conv_down(x)
        # x = self.max_pool(x)
        # x = self.relu(x)

        # x = self.conv_out(x)

        x = torch.flatten(x, 1)
        
        return x 