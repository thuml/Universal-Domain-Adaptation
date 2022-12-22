
import torch
from torch import nn
from torchvision import models

from transformers import AutoModel

from easydl import *



class MLMHead(torch.nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = hidden_size
        
        # BertPredictionHeadTransform
        # https://github.com/huggingface/transformers/blob/2c8b508ccabea6638aa463a137852ff3b64be036/src/transformers/models/bert/modeling_bert.py#L668
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.gelu = torch.nn.GELU()
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size, config.layer_norm_eps)

        # BertLMPredictionHead
        # https://github.com/huggingface/transformers/blob/2c8b508ccabea6638aa463a137852ff3b64be036/src/transformers/models/bert/modeling_bert.py#L685
        self.decoder = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(self.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, labels):
        
        # BertPredictionHeadTransform
        # shape : (batch, length, hidden_dim)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # BertLMPredictionHead
        # shape (batch, length, vocab_size)
        predictions = self.decoder(hidden_states)

        return predictions

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


class UDALM(nn.Module):
    def __init__(self, model_name, num_class, **kwargs):
        super(UDALM, self).__init__()
        print(f'INIT UDALM...')

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
        self.mlm_head = MLMHead(hidden_size=self.model.config.hidden_size, config=self.model.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        embeddings_only=False,
        is_source=True,
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

        if is_source:
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
        else:
            # masked language modeling
            # shape : (batch, length, vocab_size)
            mlm_logits = self.mlm_head(hidden_states=last_hidden_state, labels=labels)

            return {
                # shape : (batch, length, vocab_size)
                'logits' : mlm_logits,
            }

