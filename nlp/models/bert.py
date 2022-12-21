
import torch
from torch import nn
from torchvision import models

from transformers import AutoModel

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


class BERT(nn.Module):
    def __init__(self, model_name, num_class, **kwargs):
        super(BERT, self).__init__()
        print(f'INIT {model_name}...')

        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class

        try:
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.model.config.hidden_size
            
        self.classifier = CLS(self.hidden_dim, self.num_class, bottle_neck_dim=256)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        embeddings_only=False,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # shape : (batch, length, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        # shape : (batch, hidden_dim)
        cls_state = last_hidden_state[:, 0, :]

        if embeddings_only:
            return cls_state

        # cls_state         : (batch, hidden_dim)
        # after_bottlenack  : (batch, 256)
        # before_softmax    : (batch, num_class)
        # after_softmax     : (batch, num_class)
        cls_state, after_bottleneck, before_softmax, after_softmax = self.classifier(cls_state)
        
        # shape : (batch, )
        predictions = after_softmax.argmax(dim=-1)
        # shape : (batch, )
        max_logits = after_softmax.max(dim=-1).values

        return {
            'predictions' : predictions,
            'logits' : after_softmax,
            'max_logits' : max_logits
        }
