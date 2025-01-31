import torch
import torch.nn as nn
import torch.nn.functional as F

image = torch.rand(6, 6)
print("Original image:")
print(image)

image = image.unsqueeze(dim=0)  # shape: (1, 6, 6)
image = image.unsqueeze(dim=0)  # shape: (1, 1, 6, 6)
print("Image shape after adding batch and channel dimensions:", image.shape)

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)

outimage_conv2d = conv_layer(image)
print("\nOutput after convolution using nn.Conv2d:")
print(outimage_conv2d)
print("Output shape using nn.Conv2d:", outimage_conv2d.shape)

kernels = conv_layer.weight.data
print("\nKernels (weights) from nn.Conv2d:")
print(kernels)


outimage_func_conv2d = F.conv2d(image, kernels, stride=1, padding=0)
print("\nOutput after convolution using F.conv2d:")
print(outimage_func_conv2d)
print("Output shape using F.conv2d:", outimage_func_conv2d.shape)
