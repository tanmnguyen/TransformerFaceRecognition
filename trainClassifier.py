import torch
import argparse 
import torch.optim as optim

from settings import settings 
from datasets.FaceDataset import FaceDataset
from torch.utils.data import random_split, DataLoader

from utils.log import log
from utils.selection import get_encoder_from_siamese
from utils.batch import classifier_collate_fn
from utils.epoch import train_classifier_net, valid_classifier_net

from models.ClassifierNet import ClassifierNet

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

    encoder = get_encoder_from_siamese(settings.arch, args.encoder_weight)
    encoder.requires_grad_(False)

    # return
    model = ClassifierNet(encoder=encoder, num_classes=dataset.num_classes)
    model = model.to(settings.device)
    log(model)

    optimizer = optim.Adam(model.parameters(), lr=float(settings.classifier_lr), weight_decay=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.75, verbose=False)

    # train 
    for epoch in range(int(settings.classifier_epochs)):
        train_classifier_net(model, train_dataloader, optimizer, epoch)
        valid_classifier_net(model, valid_dataloader, epoch)

    # best_triplet_loss = 1e9
    # train_history, valid_history = [], []
    # for epoch in range(int(settings.epochs)):
    #     train_history.append(train_net(model, train_dataloader, optimizer, epoch))
    #     valid_history.append(valid_net(model, valid_dataloader, epoch))

    #     # save best 
    #     if valid_history[-1]["triplet_loss"] < best_triplet_loss:
    #         best_triplet_loss = valid_history[-1]["triplet_loss"]
    #         torch.save(model.state_dict(), os.path.join(f"{settings.result_path}", "best_siamese_net.pth"))

    # log(f"Best Triplet Loss: {best_triplet_loss}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    parser.add_argument('-encoder-weight',
                        '--encoder-weight',
                        required=True,
                        help="path to encoder weight .pth file")
    
    args = parser.parse_args()
    main(args)