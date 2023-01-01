
import torch
from torch import nn
from torchvision import models

from easydl import *


class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, normalize=True):
        super(ResNet50Fc, self).__init__()
        
        self.model_resnet = models.resnet50(pretrained=True)

        # pretrain model is used, use ImageNet normalization
        self.normalize = True
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
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
        return x

    def output_num(self):
        return self.__in_features


model_dict = {
    'resnet50': ResNet50Fc,
    # 'vgg16': VGG16Fc
}

class ResNet(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(ResNet, self).__init__()
        print('INIT RESNET50')
        self.feature_extractor = model_dict[args.model.base_model]()
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        return y

    def get_prediction_and_logits(self, x):
        # y : (batch, num_source_class)
        # y = self.forward(x)

        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)

        # shape : (batch, )
        predictions = y.argmax(dim=-1)
        max_logits = y.max(dim=-1).values

        return {
            'predictions' : predictions,
            'total_logits' : y,
            'max_logits' : max_logits
        }

    def get_feature(self, x):
        
        # shape : (batch, hidden_dim)
        feat = self.feature_extractor(x)

        return feat
    
