import torch 

from tqdm import tqdm
from utils.log import log 
from settings import settings 

def train_net(model, train_dataloader, optimizer, epoch):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for triplet in tqdm(train_dataloader):
        img1, img2, img3 = triplet 
        optimizer.zero_grad()
        
        # forward pass
        loss, dis_pos, dis_neg = model(img1, img2, img3)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # compute triplet accuracy 
        triplet_acc = (dis_pos < dis_neg).sum().item() / len(dis_pos)
        epoch_acc += triplet_acc    

    log(f"[Train] Epoch {epoch+1}/{settings.epochs}, Loss: {epoch_loss/len(train_dataloader)}, Acc: {epoch_acc/len(train_dataloader)}")

    return {
        "epoch": epoch,
        "triplet_loss": epoch_loss/len(train_dataloader),
        "triplet_acc": epoch_acc/len(train_dataloader)
    }

def valid_net(model, valid_dataloader, epoch):
    model.eval() 
    epoch_loss, epoch_acc = 0.0, 0.0
    with torch.no_grad():
        for triplet in tqdm(valid_dataloader):
            img1, img2, img3 = triplet 
            loss, dis_pos, dis_neg = model(img1, img2, img3)

            epoch_loss += loss.item()

            # compute triplet accuracy 
            triplet_acc = (dis_pos < dis_neg).sum().item() / len(dis_pos)
            epoch_acc += triplet_acc
    
    log(f"[Valid] Epoch {epoch+1}/{settings.epochs}, Loss: {epoch_loss/len(valid_dataloader)}, Acc: {epoch_acc/len(valid_dataloader)}")

    return {
        "epoch": epoch,
        "triplet_loss": epoch_loss/len(valid_dataloader),
        "triplet_acc": epoch_acc/len(valid_dataloader)
    }