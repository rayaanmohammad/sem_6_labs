import torch
import numpy as np

numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print(numpy_array)
print()

tensor = torch.tensor(numpy_array)
print(tensor)
print()

back_to_numpy = tensor.numpy()
print(back_to_numpy)
