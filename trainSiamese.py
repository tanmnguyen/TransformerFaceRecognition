# Train the facial encoder embedded in the Siamese network using the triplet loss.
import os 
import torch
import argparse 
import torch.optim as optim

from models.SiameseNet import SiameseNet
from models.TripletLoss import TripletLoss
from models.FaceReconstruction import FaceReconstruction

from utils.log import log
from utils.selection import get_encoder
from utils.batch import siamese_collate_fn
from utils.epoch import train_siamese_net, valid_siamese_net
from settings import settings
from torch.utils.data import DataLoader
from datasets.TripleFaceDataset import TripleFaceDataset

def main(args):
    settings.update_config(args.config)

    train_ds = TripleFaceDataset(settings.train_path)
    valid_ds = TripleFaceDataset(settings.valid_path)

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=siamese_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=siamese_collate_fn, 
        shuffle=False
    )

    # load model 
    encoder = get_encoder(settings.arch, settings.encoder_weight_path)
    model = SiameseNet(encoder=encoder, loss=TripletLoss())
    model.to(settings.device)

    log(model)
    log(f"Train set size: {len(train_ds)}")
    log(f"Valid set size: {len(valid_ds)}")
    log(f"Device: {settings.device}")
    log(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.get_param_groups(default_lr=float(settings.siamese_lr)), weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 2, gamma=0.8)

    # train 
    train_history, valid_history = [], []
    for epoch in range(int(settings.siamese_epochs)):
        train_history.append(train_siamese_net(model, train_dataloader, optimizer, lr_scheduler, epoch, settings.siamese_epochs))
        valid_history.append(valid_siamese_net(model, valid_dataloader, epoch, settings.siamese_epochs))

        # save every epoch 
        torch.save(model.state_dict(), os.path.join(f"{settings.result_path}", f"siamese_net_epoch_{epoch}.pth"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    args = parser.parse_args()
    main(args)