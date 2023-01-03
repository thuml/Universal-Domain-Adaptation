
import torch
import torch.nn.functional as F
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


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        # if option == 'resnet18':
        #     model_ft = models.resnet18(pretrained=pret)
        #     self.dim = 512
        # if option == 'resnet34':
        #     model_ft = models.resnet34(pretrained=pret)
        #     self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        # if option == 'resnet101':
        #     model_ft = models.resnet101(pretrained=pret)
        # if option == 'resnet152':
        #     model_ft = models.resnet152(pretrained=pret)

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

class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)


class OVANET(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(OVANET, self).__init__()
        print('INIT OVANET...')
        self.num_class = len(source_classes)
        self.unknown_class = self.num_class
        self.hidden_dim = 2048

        self.G = ResBase()
        self.C2 = ResClassifier_MME(num_classes=2 * self.num_class,
                           norm=False, input_size=self.hidden_dim)
        self.C1 = ResClassifier_MME(num_classes=self.num_class,
                           norm=False, input_size=self.hidden_dim)


    def forward(self, x):
        feat = self.G(x)
        out_s = self.C1(feat)
        out_open = self.C2(feat)

        return out_s, out_open
    

    def get_prediction_and_logits(self, x):
        # y : (batch, num_source_class)
        # y = self.forward(x)

        feat = self.G(x)
        # shape : (batch, num_source_class)
        out_t = self.C1(feat)

        # shape : (batch, )
        predictions = out_t.argmax(dim=-1)

        # shape : (batch, num_source_class * 2)
        out_open = self.C2(feat)
        # shape : (batch, 2, num_source_class)
        out_open = F.softmax(out_open.view(out_t.size(0), 2, -1), 1)

        # shape : (batch, )
        tmp_range = torch.range(0, out_t.size(0)-1).long().cuda()
        # shape : (batch, )
        # prob. of predicting "unknown"
        pred_unk = out_open[tmp_range, 0, predictions]
        # shape : (num_unknown, )
        # index of "unknown" samples
        ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]

        # change predictions (unknowns)
        predictions[ind_unk] = self.unknown_class

        return {
            'predictions' : predictions,
            'total_logits' : out_t,
            'max_logits' : pred_unk
        }

    def get_feature(self, x):
        
        # shape : (batch, hidden_dim)
        feat = self.G(x)

        return feat
    


        