"""
ResNet code from: 

https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/convnet-resnet18-celeba-dataparallel.ipynb
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(model_name):
  """
  returns appropriate model class based on its name
  Args:
      model_name (str): one of [small_model, celeba] 
  
  Returns:
      correct model class
  """
  model_dict = {
      'small_model': SmallClassifier,
      'ResNet18': ResNet18,
      'CNN5': Classifier_CNN5,
      'CNN3': Classifier_CNN3
  }
  return model_dict[model_name]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale=False):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # modified layer4's stride to be 1 instead of 2 bc of smaller 64x64 size
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class SmallClassifier(nn.Module):
  """
  simple classifier (e.g. for classifying MNIST digits)
  """
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1)
      self.fc1 = nn.Linear(4*4*50, 500)
      self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

class Classifier_CNN5(nn.Module):
    def __init__(self, nc, ncf, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ncf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf) x 32 x 32
            nn.Conv2d(ncf, ncf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*2) x 16 x 16
            nn.Conv2d(ncf * 2, ncf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*4) x 8 x 8
            nn.Conv2d(ncf * 4, ncf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ncf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf*8) x 4 x 4
            nn.Conv2d(ncf * 8, num_classes, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        x = self.main(input)
        logits = x.view(x.size(0), -1)
        probas = F.softmax(logits, dim=1)    
        return logits, probas
    
class Classifier_CNN3(nn.Module):
    def __init__(self, nc, ncf, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ncf, 4, 4, 1, bias=False),
            nn.BatchNorm2d(ncf),
            nn.ReLU(True),
            # state size. (ncf) x 16 x 16
            nn.Conv2d(ncf, ncf * 2, 4, 4, 1, bias=False),
            nn.BatchNorm2d(ncf * 2),
            nn.ReLU(True),
            # state size. (ncf*2) x 4 x 4
            nn.Conv2d(ncf * 2, num_classes, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        x = self.main(input)
        logits = x.view(x.size(0), -1)
        probas = F.softmax(logits, dim=1)    
        return logits, probas