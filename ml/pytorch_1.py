import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)

labels = torch.rand(1, 1000)

predict = model(data)
