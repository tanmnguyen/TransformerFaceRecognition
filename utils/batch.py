import cv2 
import torch 
from settings import settings 

def process_image(img):
    # do something with the image
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.float()
    return img

def triplet_collate_fn(batch):
    batch_img1, batch_img2, batch_img3 = [] ,[], []
    for triple_data in batch:
        img1 = cv2.imread(triple_data[0])
        img2 = cv2.imread(triple_data[1])
        img3 = cv2.imread(triple_data[2])

        # convert to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

        # process image 
        img1 = process_image(img1).to(settings.device)
        img2 = process_image(img2).to(settings.device)
        img3 = process_image(img3).to(settings.device)

        batch_img1.append(img1)
        batch_img2.append(img2)
        batch_img3.append(img3)
    
    batch_img1 = torch.stack(batch_img1)
    batch_img2 = torch.stack(batch_img2)
    batch_img3 = torch.stack(batch_img3)
    return (batch_img1, batch_img2, batch_img3)