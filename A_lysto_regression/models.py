# %%
import torch
import torchvision
from torchsummary import summary
from torch import nn

# %%
class ResNet_A(nn.Module):

  def __init__(self):
    """Ciao
    """
    super().__init__()
    self.base_net = torchvision.models.resnet101(pretrained=True)
    self.base_net.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
    )

    def forward(self, x):
      return self.base_net(x)

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class ResNet_ref_b(nn.Module):

  def __init__(self):
    super().__init__()
    mod_gen = torchvision.models.resnet50(pretrained=True).children()
    self.base_modules = nn.Sequential(*list(mod_gen)[: -2])

    self.cnn_head = nn.Sequential(
        nn.Conv2d(2048, 1024, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
    )

    self.reg_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((3,3)),
        nn.AdaptiveMaxPool2d((1,1)),
        nn.BatchNorm2d(256),
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.ReLU(inplace=True),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.Dropout(0.5),
        nn.Linear(64, 1)
    )

    self.cls_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((3,3)),
        nn.AdaptiveMaxPool2d((1,1)),
        nn.BatchNorm2d(256),
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.ReLU(inplace=True),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.Dropout(0.5),
        nn.Linear(64, 7)
    )

  def forward(self, x):

    for module in self.base_modules:
      x = module(x)
    x = self.cnn_head(x)
    reg = self.reg_head(x)
    cls_ = self.cls_head(x)
    return reg, cls_
