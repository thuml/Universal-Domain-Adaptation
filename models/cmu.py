
import torch
from torch import nn
from torchvision import models

from easydl import *


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

class ResNet50Fc(BaseFeatureExtractor):
    def __init__(self, model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        
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
    

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        # self.bottleneck_bn = nn.BatchNorm1d(bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.fc1 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc2 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc3 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc4 = nn.Linear(bottle_neck_dim, out_dim)
        self.fc5 = nn.Linear(bottle_neck_dim, out_dim)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x):
        feature = self.bottleneck(x)
        # feature = self.bottleneck_bn(feature)
        fc2_s = self.fc(feature)
        fc2_s2 = self.fc2(feature)
        fc2_s3 = self.fc3(feature)
        fc2_s4 = self.fc4(feature)
        fc2_s5 = self.fc5(feature)
        predict_prob = nn.Softmax(dim=-1)(fc2_s)

        return x, feature, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
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



class CMU(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(CMU, self).__init__()
        print('INIT CMU ....')
        self.num_class = len(source_classes)
        # pretrained resnet
        self.feature_extractor = ResNet50Fc()

        self.classifier = CLS(self.feature_extractor.output_num(), self.num_class, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)


    def forward(self, x):
        pass

        
    def get_prediction_and_logits(self, x):
        # shape : (batch, hidden_dim)
        feat = self.feature_extractor(x)

        feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = self.classifier(feat)


        # shape : (batch, )
        predictions = predict_prob.argmax(dim=-1)
        # shape : (batch, )
        max_logits = predict_prob.max(dim=-1).values


        return {
            'predictions' : predictions,
            'total_logits' : fc2_s,
            'max_logits' : max_logits
        }

    def get_feature(self, x):
        
        # shape : (batch, hidden_dim)
        feat = self.feature_extractor(x)

        return feat