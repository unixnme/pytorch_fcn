from resnet_fcn import resnet50
import torch

DEVICE = 'cuda'

model = resnet50(pretrained=True).to(DEVICE)
print(model)
i,j = 2,2
x = torch.randn(10, 3, 224*i, 224*j).to(DEVICE)
with torch.no_grad():
    y = model(x)
print(y.shape)
