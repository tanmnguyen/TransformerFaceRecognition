# import torch.nn as nn

# class UpsamplePixShuffle(nn.Module):
#     def __init__(self, upscale_factor, in_channels, out_channels):
#         super().__init__()   
#         self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=1)
#         self.relu = nn.ReLU()
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pixel_shuffle(x)
#         return x
        

# class FaceReconstruction(nn.Module):
#     def __init__(self, in_width, in_height, in_channels, out_channels, out_width, out_height):
#         super().__init__()
#         self.in_width = in_width
#         self.in_height = in_height
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.out_width = out_width
#         self.out_height = out_height
        
#         self.up1 = 

#     def forward(self, x):
#        pass



# import torch
# import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, upscale_factor, num_channels=3, base_channels=64):
#         super(Generator, self).__init__()
#         self.upscale_factor = upscale_factor
        
#         # Initial convolutional layer
#         self.initial_conv = nn.Conv2d(num_channels, base_channels, kernel_size=9, padding=4)
#         self.relu = nn.ReLU(inplace=True)
        
#         # Residual blocks
#         self.residual_blocks = self.make_residual_blocks(base_channels)
        
#         # Upsampling blocks
#         self.upsampling = self.make_upsampling_blocks(base_channels)
        
#         # Final convolutional layer
#         self.final_conv = nn.Conv2d(base_channels, num_channels * (upscale_factor ** 2), kernel_size=9, padding=4)

#         # Pixel shuffle layer
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
#     def make_residual_blocks(self, base_channels, num_blocks=16):
#         blocks = []
#         for _ in range(num_blocks):
#             blocks.append(nn.Sequential(
#                 nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(base_channels),
#                 nn.ReLU(inplace=True)
#             ))
#         return nn.Sequential(*blocks)
    
#     def make_upsampling_blocks(self, base_channels, num_blocks=2):
#         blocks = []
#         for _ in range(num_blocks):
#             blocks.append(nn.Sequential(
#                 nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
#                 nn.PixelShuffle(2),
#                 nn.ReLU(inplace=True)
#             ))
#         return nn.Sequential(*blocks)
    
#     def forward(self, x):
#         x = self.relu(self.initial_conv(x))
#         residual = x
#         x = self.residual_blocks(x)
#         x += residual
#         x = self.upsampling(x)
#         x = self.final_conv(x)
#         x = self.pixel_shuffle(x)
#         return x
