import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class FaceEncoderResnet(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.4):
        super().__init__()
        return_nodes = {
            "layer4.1.bn2": "features"
        }

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = create_feature_extractor(resnet, return_nodes=return_nodes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7 , hidden_dim)

    def get_fine_tuned_param_groups(self):
        # only return fc layer for fine tuning
        return [
            {"params": self.fc.parameters(), "lr": 1e-4}
        ]

    def forward(self, x):
        # extract features 
        x = self.cnn(x)['features']
        x = self.dropout(x)
        
        # map to latent space 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 