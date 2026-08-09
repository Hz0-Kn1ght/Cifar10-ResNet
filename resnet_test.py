import torch
from models.ResNet18 import ResNet

model = ResNet()

x = torch.randn(4,3,32,32)

y = model(x)

print(y.shape)