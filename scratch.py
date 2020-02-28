import torch

data = torch.tensor([0,1,2]).expand(3,3)
data = data[1:,:]
print(data)


