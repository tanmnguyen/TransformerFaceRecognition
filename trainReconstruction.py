# Train the facial encoder embedded in the Siamese network using the triplet loss.
import os 
import torch
import argparse 
import torch.optim as optim

from models.FaceReconstruction import FaceReconstruction

from utils.log import log
from utils.batch import classifier_collate_fn
from utils.selection import get_encoder_from_siamese
from torch.utils.data import random_split, DataLoader
from utils.epoch import train_recon_net, valid_recon_net

from settings import settings
from torch.utils.data import DataLoader
from datasets.FaceDataset import FaceDataset

def main(args):
    settings.update_config(args.config)

    dataset = FaceDataset(settings.train_path)
    train_ds, valid_ds = random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=classifier_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_ds, 
        batch_size=int(settings.batch_size), 
        collate_fn=classifier_collate_fn, 
        shuffle=False
    )

    # load model 
    encoder = get_encoder_from_siamese(settings.arch, settings.encoder_weight_path)
    encoder = encoder.to(settings.device)
    encoder.requires_grad_(False)

    # reconstruction model
    b, enc_len = encoder(torch.randn(1, 3, 224, 224)).shape
    model = FaceReconstruction(enc_len, 7, 7, 512, 3, 224, 224).to(settings.device)

    log(model)
    log(f"Train set size: {len(train_ds)}")
    log(f"Valid set size: {len(valid_ds)}")
    log(f"Device: {settings.device}")
    log(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) * 2, gamma=0.8)

    # train 
    train_history, valid_history = [], []
    for epoch in range(int(settings.siamese_epochs)):
        train_history.append(train_recon_net(model, encoder, train_dataloader, optimizer, lr_scheduler, epoch, settings.siamese_epochs))
        valid_history.append(valid_recon_net(model, encoder, valid_dataloader, epoch, settings.siamese_epochs))

        # save every epoch 
        torch.save(model.state_dict(), os.path.join(f"{settings.result_path}", f"recon_model_{epoch}.pth"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    args = parser.parse_args()
    main(args)