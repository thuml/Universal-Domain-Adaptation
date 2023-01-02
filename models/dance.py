
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import models

from easydl import *



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd=ctx.saved_tensors[0]
        return grad_output.neg()*lambd, None
    
def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x, Variable(torch.ones(1)*lambd).cuda())


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, unit_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        return x

class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05):
        super(ResClassifier_MME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)
        x = self.fc(x)/self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)


class DANCE(nn.Module):
    def __init__(self, args, source_classes, **kwargs):
        super(DANCE, self).__init__()
        print('INIT DANCE...')
        self.num_class = len(source_classes)
        self.hidden_dim = 2048
        self.G = ResBase(unit_size=self.hidden_dim)
        self.C = ResClassifier_MME(num_classes=self.num_class, input_size=self.hidden_dim, temp=0.5)
