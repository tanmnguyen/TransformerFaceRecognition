import torch 

from tqdm import tqdm
from utils.log import log 
from settings import settings 
from torchmetrics import Accuracy

def train_classifier_net(model, train_dataloader, optimizer, epoch, total_epochs):
    model.train() 
    acc_func = Accuracy(task="multiclass", num_classes=model.num_classes).to(settings.device)

    epoch_loss, epoch_acc = 0.0, 0.0
    for img, lbl in tqdm(train_dataloader):
        optimizer.zero_grad()
        output = model(img)
        loss = model.loss(output, lbl)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc_func(output, lbl)

    log(f"[Train] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(train_dataloader)}, Acc: {epoch_acc/len(train_dataloader)}")

def valid_classifier_net(model, valid_dataloader, epoch):  
    pass

def train_recon_net(model, encoder, train_dataloader, optimizer, scheduler, epoch, total_epochs):
    model.train() 

    epoch_loss = 0.0
    for i, img, _ in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        with torch.no_grad():
            latent = encoder(img)
        output = model(latent)
        loss = model.loss(output, img)

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            epoch_loss += loss.item()

        if i % 50 == 0:
            log(f"[Train] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(train_dataloader)}")

    log(f"[Train] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(train_dataloader)}")
    return {
        "epoch": epoch,
        "recon_loss": epoch_loss/len(train_dataloader)
    }

def valid_recon_net(model, encoder, valid_dataloader, epoch, total_epochs):
    model.eval() 

    epoch_loss = 0.0
    with torch.no_grad():
        for img, _ in tqdm(valid_dataloader):
            with torch.no_grad():
                latent = encoder(img)
            output = model(latent)
            loss = model.loss(output, img)

            epoch_loss += loss.item()

    log(f"[Valid] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(valid_dataloader)}")

    return {
        "epoch": epoch,
        "recon_loss": epoch_loss/len(valid_dataloader)
    }

def train_siamese_net(model, train_dataloader, optimizer, scheduler, epoch, total_epochs):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for i, triplet in enumerate(tqdm(train_dataloader)):
        img1, img2, img3 = triplet 
        optimizer.zero_grad()
        
        # forward pass
        loss, dis_pos, dis_neg = model(img1, img2, img3)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        # compute triplet accuracy 
        triplet_acc = (dis_pos < dis_neg).sum().item() / len(dis_pos)
        epoch_acc += triplet_acc    

        if i % 100 == 0:
            log(f"[Train] Epoch {epoch+1}/{total_epochs}: " + \
                f"Batch {i}/{len(train_dataloader)} " +\
                f"| Loss: {epoch_loss / (i + 1)} " +\
                f"| Acc: {epoch_acc / (i + 1)} " + \
                f"| LR: {scheduler.get_last_lr()[0]}"
            )


    log(f"[Train] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(train_dataloader)}, Acc: {epoch_acc/len(train_dataloader)}")

    return {
        "epoch": epoch,
        "triplet_loss": epoch_loss/len(train_dataloader),
        "triplet_acc": epoch_acc/len(train_dataloader)
    }

def valid_siamese_net(model, valid_dataloader, epoch, total_epochs):
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
    
    log(f"[Valid] Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss/len(valid_dataloader)}, Acc: {epoch_acc/len(valid_dataloader)}")

    return {
        "epoch": epoch,
        "triplet_loss": epoch_loss/len(valid_dataloader),
        "triplet_acc": epoch_acc/len(valid_dataloader)
    }