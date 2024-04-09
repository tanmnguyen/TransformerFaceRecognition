import sys
sys.path.append('../')

import settings 
import torch.nn as nn

from .DAT.dat import DAT

class FaceEncoderDat(nn.Module):
    def __init__(self,  
            img_size = 224,
            patch_size = 4,
            num_classes = 1000, 
            expansion = 4,
            dim_stem = 128, 
            dims = [64, 128, 256, 512],
            depths = [2, 2, 2, 2],
            stage_spec = [
                ['N', 'D'], 
                ['N', 'D'], 
                ['N', 'D'], 
                ['D', 'D']
            ],
            heads = [4, 8, 16, 32],
            window_sizes = [7, 7, 7, 7],
            groups = [2, 4, 8, 16],
            use_pes = [True, True, True, True],
            dwc_pes = [False, False, False, False],
            strides = [8, 4, 2, 1],
            offset_range_factor = [-1, -1, -1, -1],
            no_offs = [False, False, False, False],
            fixed_pes = [False, False, False, False],
            use_dwc_mlps = [True, True, True, True],
            use_lpus = [True, True, True, True],
            use_conv_patches = True,
            ksizes = [9, 7, 5, 3],
            nat_ksizes = [7, 7, 7, 7],
            drop_rate = 0.0,
            attn_drop_rate = 0.0,
            drop_path_rate = 0.7,
            hidden_dim=512,
    ):
        super().__init__()
        self.dat = DAT(
            img_size = img_size,
            patch_size = patch_size,
            num_classes = hidden_dim, # hidden dim
            expansion = expansion,
            dim_stem = dim_stem, 
            dims = dims,
            depths = depths,
            stage_spec = stage_spec,
            heads = heads,
            window_sizes = window_sizes,
            groups = groups,
            use_pes = use_pes,
            dwc_pes = dwc_pes,
            strides = strides,
            offset_range_factor = offset_range_factor,
            no_offs = no_offs,
            fixed_pes = fixed_pes,
            use_dwc_mlps = use_dwc_mlps,
            use_lpus = use_lpus,
            use_conv_patches = use_conv_patches,
            ksizes = ksizes,
            nat_ksizes = nat_ksizes,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate
        )

    def forward(self, x):
        x = self.dat(x)
        return x