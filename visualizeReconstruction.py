import cv2 
import torch
import argparse 
import numpy as np 

from utils.batch import process_image
from models.TripletLoss import TripletLoss
from utils.selection import get_encoder_from_siamese
from models.FaceReconstruction import FaceReconstruction
from models.SiameseNetFaceRecon import SiameseNetFaceRecon

from settings import settings

def main(args):
    settings.update_config(args.config)

    # test_ds = TripleFaceDataset(settings.test_path)
    # test_dataloader = DataLoader(
    #     test_ds, 
    #     batch_size=int(settings.batch_size), 
    #     collate_fn=siamese_collate_fn, 
    #     shuffle=True
    # )

    #load model 
    encoder = get_encoder_from_siamese(settings.arch, None)
    encoder = encoder.to(settings.device)
    with torch.no_grad():
        _, enc_len = encoder(torch.randn(1, 3, 224, 224).to(settings.device)).shape
    face_recon = FaceReconstruction(enc_len, 7, 7, 512, 3, 224, 224).to(settings.device)
    model = SiameseNetFaceRecon(encoder=encoder, loss=TripletLoss(), face_recon=face_recon)
    model.to(settings.device)

    # load weight 
    model.load_state_dict(torch.load(args.weight, map_location=settings.device))

    # load image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = process_image(img).to(settings.device)

    conv_img = img.permute(1, 2, 0)
    conv_img = conv_img.cpu().numpy()
    conv_img = np.uint8(conv_img)
    # cv2.imshow("Original Face", cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    img = img.unsqueeze(0)

    with torch.no_grad():
        latent = model.encoder(img)
        face_recon = model.face_recon(latent)

    face_recon = face_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    face_recon = np.uint8(face_recon)
    face_recon = cv2.cvtColor(face_recon, cv2.COLOR_RGB2BGR)
    cv2.imshow("Reconstructed Face", face_recon)
    cv2.waitKey(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config file")
    
    parser.add_argument('-image',
                        '--image',
                        required=True,
                        help="path to a face image file")
    
    parser.add_argument('-weight',
                        '--weight',
                        required=True,
                        help="path to Siamese model weight")

    args = parser.parse_args()
    main(args)