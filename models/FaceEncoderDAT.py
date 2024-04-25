import sys
sys.path.append('../')

import torch 
import torch.nn as nn

from .DAT.dat import DAT

def load_state_dict(model, weight_path):
    """
    Load a model's state dictionary while handling size mismatches.

    Parameters:
        model (torch.nn.Module): The model to load the state dictionary into.
        weight_path (str): The path to the saved state dictionary.
    """
    # Load the state dictionary while ignoring size mismatches
    try:
        loaded_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']
    except KeyError:
        loaded_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

    # Get the current model state dictionary
    current_state_dict = model.state_dict()

    # Update the current state dictionary with parameters from the loaded state dictionary,
    # while ignoring any size mismatches
    for name, param in loaded_state_dict.items():
        if name in current_state_dict:
            if param.size() != current_state_dict[name].size():
                print(f"Ignoring size mismatch for parameter '{name}'")
                continue
            current_state_dict[name].copy_(param)
        else:
            print(f"Ignoring parameter {name} because it is not in the model's state dictionary")

    # Load the updated state dictionary into the model
    model.load_state_dict(current_state_dict)

    return model 

"""
Default is DAT tiny model 
"""
class FaceEncoderDat(nn.Module):
    def __init__(self,  
            img_size=224,
            patch_size=4,
            num_classes=512 * 7 * 7,
            expansion=4,
            dim_stem=64,
            dims=[64, 128, 256, 512],
            depths=[2, 4, 18, 2],
            stage_spec=[
                ['N', 'D'], 
                ['N', 'D', 'N', 'D'], 
                ['N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D'], 
                ['D', 'D']],
            heads=[2, 4, 8, 16],
            window_sizes=[7, 7, 7, 7],
            groups=[1, 2, 4, 8],
            use_pes=[True, True, True, True],
            dwc_pes=[False, False, False, False],
            strides=[8, 4, 2, 1],
            offset_range_factor=[-1, -1, -1, -1],
            no_offs=[False, False, False, False],
            fixed_pes=[False, False, False, False],
            use_dwc_mlps=[True, True, True, True],
            use_lpus=[True, True, True, True],
            use_conv_patches=True,
            ksizes=[9, 7, 5, 3],
            nat_ksizes=[7, 7, 7, 7],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            encoder_weight_path=None
    ):
        super().__init__()
        self.dat = DAT(
            img_size = img_size,
            patch_size = patch_size,
            num_classes=num_classes,
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

        # load pretrained weight 
        if encoder_weight_path is not None:
            self.dat = load_state_dict(self.dat, encoder_weight_path)

    def get_fine_tuned_param_groups(self):
        return [
            {"params": self.dat.stages[-1].parameters(), "lr": 1e-4}
        ]
    
    def forward(self, x):
        # extract features
        features = self.dat.extract_features(x)

        # map to latent feature 
        x = torch.flatten(features, 1)
        
        return x