import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceReconstruction(nn.Module):
    def __init__(self, in_width, in_height, in_channels, out_channels, out_width, out_height):
        super().__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_width = out_width
        self.out_height = out_height

        # Fully connected layer to adjust input vector to suitable size
        self.fc = nn.Linear(in_channels * in_width * in_height, in_channels * in_width * in_height)

        # Reshape layer to go from flat vector to (b, 512, 7, 7)
        self.reshape = lambda x: x.view(-1, in_channels, in_height, in_width)

        # Transposed Convolution layers
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)  # output: (b, 256, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)          # output: (b, 128, 28, 28)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)           # output: (b, 64, 56, 56)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)            # output: (b, 32, 112, 112)
        self.deconv5 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # output: (b, out_channels, out_width, out_height)

        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.relu(self.batchnorm1(self.deconv1(x)))
        x = self.relu(self.batchnorm2(self.deconv2(x)))
        x = self.relu(self.batchnorm3(self.deconv3(x)))
        x = self.relu(self.batchnorm4(self.deconv4(x)))
        x = self.deconv5(x)  # No activation, assuming this is the output layer for an image
        return x

# Example usage
# model = FaceReconstruction(7, 7, 512, 3, 224, 224)
# x = torch.randn(1, 7 * 7 * 512)
# output = model(x)
# print(output.shape)