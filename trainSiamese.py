# Train the facial encoder embedded in the Siamese network using the triplet loss.
import os 
import torch
import argparse 
import torch.optim as optim

from models.SiameseNet import SiameseNet
from models.TripletLoss import TripletLoss
from models.FaceEncoderResnet import FaceEncoderResnet

from utils.log import log
from utils.batch import triplet_collate_fn
from utils.epoch import train_net, valid_net

from settings import settings
from torch.utils.data import random_split, DataLoader
from datasets.TripleFaceDataset import TripleFaceDataset

def main(args):
    settings.update_config(args.config)

    dataset = TripleFaceDataset(settings.train_path)
    train_ds, valid_ds = random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=triplet_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=triplet_collate_fn, 
        shuffle=False
    )

    # load model 
    if settings.arch == "resnet18":
        encoder = FaceEncoderResnet
    
    model = SiameseNet(encoder=encoder(), loss=TripletLoss())
    model.to(settings.device)
    log(model)

    optimizer = optim.Adam(model.parameters(), lr=float(settings.lr), weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.75, verbose=False)

    # train 
    best_triplet_loss = 1e9
    train_history, valid_history = [], []
    for epoch in range(int(settings.epochs)):
        train_history.append(train_net(model, train_dataloader, optimizer, epoch))
        valid_history.append(valid_net(model, valid_dataloader, epoch))

        # save best 
        if valid_history[-1]["triplet_loss"] < best_triplet_loss:
            best_triplet_loss = valid_history[-1]["triplet_loss"]
            torch.save(model.state_dict(), os.path.join(f"{settings.result_path}", "best_siamese_net.pth"))

    log(f"Best Triplet Loss: {best_triplet_loss}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    args = parser.parse_args()
    main(args)