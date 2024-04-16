import sys 
sys.path.append('../')

import torch 
from models.SiameseNet import SiameseNet
from models.TripletLoss import TripletLoss

def get_encoder_from_siamese(arch: str, weight: str):
    if arch == "resnet18":
        from models.FaceEncoderResnet import FaceEncoderResnet
        encoder = FaceEncoderResnet
    else:
        raise ValueError(f"arch {arch} not supported")

    model = SiameseNet(encoder=encoder(), loss=TripletLoss())
    model.load_state_dict(torch.load(weight))

    model = model.encoder
    return model

def get_encoder(arch: str, weight: str):
    if arch == "resnet18":
        from models.FaceEncoderResnet import FaceEncoderResnet
        encoder = FaceEncoderResnet()
    elif arch == "dat":
        from models.FaceEncoderDAT import FaceEncoderDat
        if "tiny" in weight:
            encoder = FaceEncoderDat(
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
                encoder_weight_path=weight,
            )
        
        if "base" in weight:
            encoder = FaceEncoderDat(
                img_size=224,
                patch_size=4,
                num_classes=1000,
                expansion=4,
                dim_stem=128,
                dims=[128, 256, 512, 1024],
                depths=[2, 4, 18, 2],
                stage_spec=[
                    ['N', 'D'], 
                    ['N', 'D', 'N', 'D'], 
                    ['N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D'], 
                    ['D', 'D']
                ],
                heads=[4, 8, 16, 32],
                window_sizes=[7, 7, 7, 7],
                groups=[2, 4, 8, 16],
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
                drop_path_rate=0.7,
                encoder_weight_path=weight,
            )
    else:
        raise ValueError(f"arch {arch} not supported")
    
    return encoder