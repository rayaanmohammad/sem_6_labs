import torch
tensor = torch.tensor([[ 1,  2,  3,  4],
          [ 5,  6,  7,  8],
          [ 9, 10, 11, 12]])
print(tensor[1, 2])

print(tensor[0,0:3])
