# TransformerFaceRecognition
Implementation of the paper "Neighborhood Deformable Attention Transformer for Face Recognition with Reconstruction Network". This face recognition module incorporated transformer-based encoder network to extract relevant features and use reconstruction network to guild the encoder to learn representative features of the actual face.

The network is performed on RGB image 
## Training 
Train Siamese model 
```bash 
CUDA_VISIBLE_DEVICES=0 python trainSiamese.py --config configs/CelebA-resnet.cfg
```
Train neighborhood attention deformable model 
```bash 
CUDA_VISIBLE_DEVICES=1 python trainSiamese.py --config configs/CelebA-dat-tiny.cfg
CUDA_VISIBLE_DEVICES=1 python trainSiamese.py --config configs/CelebA-dat-base.cfg
```
Train resnet neighborhood attention deformable model 
```bash 
CUDA_VISIBLE_DEVICES=0 python trainSiamese.py --config configs/CelebA-resnet_dat.cfg
```
Train classifier model
```bash
python trainClassifier.py --config configs/CelebA.cfg --encoder-weight /Users/tan/Desktop/TransformerFaceRecognition/results/resnet18-2024-03-20-00-18-11/best_siamese_net.pth
```

## Train reconstruction net 
```bash 
CUDA_VISIBLE_DEVICES=0 python trainReconstruction.py --config configs/CelebA-dat-tiny-recon.cfg
```

```bash 
CUDA_VISIBLE_DEVICES=0 python trainReconstruction.py --config configs/CelebA-resnet_dat-recon.cfg
```

```bash 
CUDA_VISIBLE_DEVICES=0 python trainReconstruction.py --config configs/CelebA-resnet-recon.cfg
```