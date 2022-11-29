
import torch

from torch import nn
import torch.nn.functional as F
from torchvision import models

from easydl import *

# https://github.com/Solacex/Domain-Consensus-Clustering/blob/7e9712cca81856bd4767dd3b0c63662f6845c383/model/res50.py#L6
class CLS(nn.Module):
    """
    From: https://github.com/thuml/Universal-Domain-Adaptation
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.bn = nn.BatchNorm1d(bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        x = self.bn(self.bottleneck(x))
        out.append(x)
        x =self.fc(x)
        out.append(x)
        out.append(F.softmax(x, dim=-1))
        return 

# https://github.com/Solacex/Domain-Consensus-Clustering/blob/7e9712cca81856bd4767dd3b0c63662f6845c383/model/res50.py#L31
class DCC(nn.Module):
    def __init__(self, args, source_classes, bottleneck=True, pretrained=True):
        super(DCC, self).__init__()

        self.num_classes = len(source_classes)
        self.hidden_dim = 2048
        self.bottleneck = bottleneck
        features = models.resnet50(pretrained=pretrained)
        self.features =  nn.Sequential(*list(features.children())[:-1])
        # classifier w/ bottleneck
        self.classifer = CLS(self.hidden_dim, self.num_classes)

    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        feat = self.features(x)
        feat = feat.squeeze()
        # classifier w/ bottleneck 
        # bottlenack : (batch, 256) 
        # prob       : (batch, num_source_class)
        # af_softmax : (batch, num_source_class)
        _, bottleneck, prob, af_softmax = self.classifer(feat)

        return feat, bottleneck, prob, F.softmax(prob, dim=-1)

    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.classifer.parameters(), 'lr':  lr*10}]
        return 
  
