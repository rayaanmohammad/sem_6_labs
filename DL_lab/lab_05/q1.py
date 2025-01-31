import torch
import torch.nn.functional as F

# Create a random image of shape (6, 6) - single channel (grayscale)
image = torch.rand(6, 6)
print("Original image:")
print(image)

# Add a batch dimension and channel dimension (4D tensor: batch_size, channels, height, width)
image = image.unsqueeze(dim=0)  # shape: (1, 6, 6)
image = image.unsqueeze(dim=0)  # shape: (1, 1, 6, 6)
print("Image shape after adding batch and channel dimensions:", image.shape)

kernel = torch.ones(3, 3)
print("Kernel shape:", kernel.shape)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)

outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("\nOutput with stride=1 and padding=0:")
print(outimage)
print("Output shape:", outimage.shape)

outimage_padding1 = F.conv2d(image, kernel, stride=1, padding=1)
print("\nOutput with stride=1 and padding=1:")
print(outimage_padding1)
print("Output shape:", outimage_padding1.shape)

