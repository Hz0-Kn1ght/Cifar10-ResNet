import torch
from dataset import get_dataloader
from model import CNN
from train import train

train_loader, test_loader = get_dataloader(64)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(labels[:10])

model = CNN()
x = torch.randn(64, 3, 32, 32)
y = model(x)
print(y.shape)

train(model, train_loader)