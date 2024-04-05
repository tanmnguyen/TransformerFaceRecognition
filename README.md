# TransformerFaceRecognition
Implementation of the paper "Neighborhood Deformable Attention Transformer for Face Recognition with Reconstruction Network". This face recognition module incorporated transformer-based encoder network to extract relevant features and use reconstruction network to guild the encoder to learn representative features of the actual face.


## Training 
Train Siamese model 
```bash 
python trainSiamese.py --config configs/CelebA.cfg
```

Train classifier model
```bash
python trainClassifier.py --config configs/CelebA.cfg --encoder-weight /Users/tan/Desktop/TransformerFaceRecognition/results/resnet18-2024-03-20-00-18-11/best_siamese_net.pth
```

