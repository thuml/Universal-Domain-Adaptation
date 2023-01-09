
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from transformers import AutoModel

from easydl import *



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
    def __init__(self, model_name, num_class, **kwargs):
        super(OVANET, self).__init__()
        print('INIT OVANET...')
        
        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class

        try:
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.model.config.hidden_size

        self.C1 = ResClassifier_MME(num_classes=self.num_class,
                           norm=False, input_size=self.hidden_dim)
        self.C2 = ResClassifier_MME(num_classes=2 * self.num_class,
                           norm=False, input_size=self.hidden_dim)


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
        batch_size, input_length = input_ids.shape

        # shape : (batch, length, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        # shape : (batch, hidden_dim)
        cls_state = last_hidden_state[:, 0, :]

        if embeddings_only:
            return cls_state
        
        # shape : (batch, num_source_class)
        out = self.C1(cls_state)
        out = F.softmax(out, 1)
        # shape : (batch, num_source_class * 2)
        out_open = self.C2(cls_state)

        predictions = out.argmax(dim=-1)
        # shape : (batch, )
        max_logits = out.max(dim=-1).values

        # shape : (batch, 2, num_source_class)
        out_open_reshaped = F.softmax(out_open.view(batch_size, 2, -1), 1)

        # shape : (batch, )
        tmp_range = torch.arange(0, batch_size).long().cuda()
        # shape : (batch, )
        # prob. of predicting "unknown"
        pred_unk = out_open_reshaped[tmp_range, 0, predictions]
        # shape : (num_unknown, )
        # index of "unknown" samples
        ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]

        # change predictions (unknowns)
        predictions[ind_unk] = self.unk_index

        return {
            'predictions' : predictions,
            'logits' : out,
            'logits_open' : out_open,
            # 'max_logits' : pred_unk
            'max_logits' : max_logits
        }
    
