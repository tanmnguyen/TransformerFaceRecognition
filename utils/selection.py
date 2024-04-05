import sys 
sys.path.append('../')

import torch 
from models.SiameseNet import SiameseNet
from models.TripletLoss import TripletLoss

def get_encoder(arch: str, weight: str):
    if arch == "resnet18":
        from models.FaceEncoderResnet import FaceEncoderResnet
        encoder = FaceEncoderResnet
    else:
        raise ValueError(f"arch {arch} not supported")

    model = SiameseNet(encoder=encoder(), loss=TripletLoss())
    model.load_state_dict(torch.load(weight))

    model = model.encoder
    return model