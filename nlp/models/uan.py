
import torch
from torch import nn

from transformers import AutoModel

from easydl import *


class CLS(nn.Module):
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


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature, max_train_step):
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
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=max_train_step))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y

class UAN(nn.Module):
    def __init__(self, model_name, num_class, max_train_step, **kwargs):
        super(UAN, self).__init__()
        print('INIT UAN...')
        
        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class
        self.max_train_step = max_train_step

        try:
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.model.config.hidden_size

        self.classifier = CLS(self.hidden_dim, self.num_class, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(self.hidden_dim, self.max_train_step)
        self.discriminator_separate = AdversarialNetwork(self.hidden_dim, self.max_train_step)

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
        
        # shape : (batch, 1)
        # d = self.discriminator(after_bottleneck)
        d = self.discriminator(cls_state)
        # shape : (batch, 1)
        # d_0 = self.discriminator_separate(after_bottleneck)
        d_0 = self.discriminator_separate(cls_state)
        
        # shape : (batch, )
        predictions = after_softmax.argmax(dim=-1)
        # shape : (batch, )
        max_logits = after_softmax.max(dim=-1).values

        target_share_weight = self.get_target_share_weight(d_0, before_softmax, domain_temperature=1.0,
                                                        class_temperature=1.0)
        


        return {
            'predictions' : predictions,
            'before_softmax' : before_softmax,
            'logits' : after_softmax,
            # 'max_logits' : target_share_weight.reshape(-1),
            'max_logits' : max_logits,
            # from discriminator
            'd' : d,
            # from discriminator_separate
            'd_0' : d_0,
        }
        
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
