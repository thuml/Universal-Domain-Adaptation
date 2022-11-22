
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from easydl import *


class Classifier(nn.Module):

    def __init__(self, out_features, num_classes,bottleneck_dim=256):
        super(Classifier, self).__init__()

        assert bottleneck_dim > 0
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(out_features, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(self.bottleneck_dim, num_classes)

    def forward(self, x):
        f = self.bottleneck(x)
        predictions = self.head(f)
        return predictions

# adversarial network
class AdversarialNetwork(nn.Module):
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

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)

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



class DANN(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(DANN, self).__init__()
        self.num_class = len(source_classes)
        self.unknown_class = self.num_class
        self.hidden_dim = 2048

        self.base_model = ResBase()
        self.classifier = Classifier(out_features=self.hidden_dim, num_classes=self.num_class)
        self.domain_discriminator = AdversarialNetwork(in_feature=self.hidden_dim)


    def forward(self, x):
        # shape : (batch, hidden_dim)
        feat = self.base_model(x)
        # shape : (batch, num_class)
        classification_output = self.classifier(feat)
        # shape : (batch, 1)
        domain_output = self.domain_discriminator(feat)

        return classification_output, domain_output

    def get_prediction_and_logits(self, x):
        # shape : (batch, hidden_dim)
        feat = self.base_model(x)
        # shape : (batch, num_class)
        classification_output = self.classifier(feat)

        # shape : (batch, )
        predictions = classification_output.argmax(dim=-1)
        # shape : (batch, )
        max_logits = classification_output.max(dim=-1).values


        return {
            'predictions' : predictions,
            'total_logits' : classification_output,
            'max_logits' : max_logits
        }
    


        