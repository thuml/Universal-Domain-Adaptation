
import torch
from torch import nn
from torchvision import models

from easydl import *



class ResNet(nn.Module):
    def __init__(self, top=False):
        super(ResNet, self).__init__()
        self.dim = 2048
        self.top = top
        # load pretrained resnet
        model_ft = models.resnet50(pretrained=True)

        # remove final layer
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)


    def forward(self, x):
        x = self.features(x)
        if self.top:
            return x
        else:
            x = x.view(x.size(0), self.dim)
            return x
        
# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/415fbe2fa3a6cb8ef858d991182ccd9ca1ed8960/new/model.py#L168
class ClassifierBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: nn.Module, bottleneck_dim: int = 256):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        self.bottleneck = bottleneck
        assert bottleneck_dim > 0
        self._features_dim = bottleneck_dim

        self.head = nn.Linear(self._features_dim, num_classes)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.dim)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params
    
# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/415fbe2fa3a6cb8ef858d991182ccd9ca1ed8960/new/model.py#L213
class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)


# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/415fbe2fa3a6cb8ef858d991182ccd9ca1ed8960/new/model.py#L65~
class DomainDiscriminator(nn.Module):

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/415fbe2fa3a6cb8ef858d991182ccd9ca1ed8960/new/model.py#L223
class Ensemble(nn.Module):

    def __init__(self, in_feature, num_classes):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(in_feature, num_classes)
        self.fc2 = nn.Linear(in_feature, num_classes)
        self.fc3 = nn.Linear(in_feature, num_classes)
        self.fc4 = nn.Linear(in_feature, num_classes)
        self.fc5 = nn.Linear(in_feature, num_classes)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x, index=0):
        if index == 1:
            y = self.fc1(x)
            # y = nn.Softmax(dim=-1)(y_1)
        elif index == 2:
            y = self.fc2(x)
            # y = nn.Softmax(dim=-1)(y_2)
        elif index == 3:
            y = self.fc3(x)
            # y = nn.Softmax(dim=-1)(y_3)
        elif index == 4:
            y = self.fc4(x)
            # y = nn.Softmax(dim=-1)(y_4)
        elif index == 5:
            y = self.fc5(x)
            # y = nn.Softmax(dim=-1)(y_5)
        else:
            y_1 = self.fc1(x)
            y_1 = nn.Softmax(dim=-1)(y_1)
            y_2 = self.fc2(x)
            y_2 = nn.Softmax(dim=-1)(y_2)
            y_3 = self.fc3(x)
            y_3 = nn.Softmax(dim=-1)(y_3)
            y_4 = self.fc4(x)
            y_4 = nn.Softmax(dim=-1)(y_4)
            y_5 = self.fc5(x)
            y_5 = nn.Softmax(dim=-1)(y_5)
            return y_1, y_2, y_3, y_4, y_5

        return y

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.parameters(), "lr_mult": 1.},
        ]
        return params
    


class CMU(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(CMU, self).__init__()
        print('INIT CMU')
        self.num_class = len(source_classes)
        # pretrained resnet
        self.backbone = ResNet()

        self.classifier = ImageClassifier(backbone=self.backbone, num_classes=self.num_class)
        self.domain_discriminator = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=1024)
        self.ensemble = Ensemble(in_feature=self.classifier.features_dim, num_classes=self.num_class)

    def forward(self, x, index=0):
        # classifier_prediction : (batch, num_source_class)
        # f                     : (batch, 256)
        classifier_prediction, f = self.classifier(x)

        # import pdb
        # pdb.set_trace()

        ensemble_prediction = self.ensemble(f, index=index)

        return classifier_prediction, f, ensemble_prediction
