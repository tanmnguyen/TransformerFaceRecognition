# Train the facial encoder embedded in the Siamese network using the triplet loss.
import os 
import torch
import argparse 

from models.SiameseNet import SiameseNet
from models.TripletLoss import TripletLoss

from utils.log import log
from utils.selection import get_encoder
from utils.epoch import valid_siamese_net
from utils.batch import siamese_collate_fn

from settings import settings
from torch.utils.data import DataLoader
from datasets.TripleFaceDataset import TripleFaceDataset

from utils.selection import get_encoder_from_siamese
from models.FaceReconstruction import FaceReconstruction
from models.SiameseNetFaceRecon import SiameseNetFaceRecon

def main(args):
    settings.update_config(args.config)

    test_ds = TripleFaceDataset(settings.test_path)
    test_dataloader = DataLoader(
        test_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=siamese_collate_fn, 
        shuffle=True
    )

    # load model 
    try:
        encoder = get_encoder(settings.arch, settings.encoder_weight_path)
        model = SiameseNet(encoder=encoder, loss=TripletLoss())
        model.to(settings.device)

        # try if the weight can be loaded
        model.load_state_dict(torch.load(
                os.path.join(args.checkpoint, f"siamese_net_epoch_{0}.pth"), 
                map_location=settings.device
            )
        )
    except:
        encoder = get_encoder_from_siamese(settings.arch, None)
        encoder = encoder.to(settings.device)
        with torch.no_grad():
            _, enc_len = encoder(torch.randn(1, 3, 224, 224).to(settings.device)).shape
        face_recon = FaceReconstruction(enc_len, 7, 7, 512, 3, 224, 224).to(settings.device)
        model = SiameseNetFaceRecon(encoder=encoder, loss=TripletLoss(), face_recon=face_recon)
        model.to(settings.device)

    log(model)
    log(f"Test set size: {len(test_ds)}")
    log(f"Device: {settings.device}")
    log(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(int(settings.siamese_epochs)):
        # load weight 
        model.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint, f"siamese_net_epoch_{epoch}.pth"), 
                map_location=settings.device
            )
        )
        valid_siamese_net(model, test_dataloader, epoch, settings.siamese_epochs)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    parser.add_argument('-checkpoint',
                        '--checkpoint',
                        required=True,
                        help="path to checkpoint directory")

    
    args = parser.parse_args()
    main(args)