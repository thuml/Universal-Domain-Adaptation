
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


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y



model_dict = {
    'resnet50': ResNet50Fc,
    # 'vgg16': VGG16Fc
}

class UAN(nn.Module):
    def __init__(self, args, source_classes):
        super(UAN, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model]()
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0

        
    def reverse_sigmoid(self, y):
        return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

    def get_source_share_weight(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
        before_softmax = before_softmax / class_temperature
        after_softmax = nn.Softmax(-1)(before_softmax)
        domain_logit = self.reverse_sigmoid(domain_out)
        domain_logit = domain_logit / domain_temperature
        domain_out = nn.Sigmoid()(domain_logit)
        
        entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        weight = entropy_norm - domain_out
        weight = weight.detach()
        return weight


    def get_target_share_weight(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
        return - self.get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


    def normalize_weight(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        x = x / torch.mean(x)
        return x.detach()

