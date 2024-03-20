import torch.nn as nn
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

class FaceEncoderResnet(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        return_nodes = {
            "layer4.1.bn2": "features"
        }

        resnet = resnet18(pretrained=True)
        self.cnn = create_feature_extractor(resnet, return_nodes=return_nodes)

        self.fc = nn.Linear(512 * 7 * 6 , hidden_dim)

    def forward(self, x):
        # extract features 
        x = self.cnn(x)['features']

        # map to latent space 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 