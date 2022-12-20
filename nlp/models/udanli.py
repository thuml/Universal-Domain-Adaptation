
from torch import nn
from torch.autograd import Function

from transformers import AutoModel

from easydl import *


# can be replaced with GradientReverseModule
# from easydl import GradientReverseModule
# CODE FROM : https://github.com/fungtion/DANN/blob/master/models/functions.py
class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output

# classification head for NLI
class Classifier(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.bottle_neck_dim = in_dim // 2
        self.bottleneck = nn.Linear(in_dim, self.bottle_neck_dim)
        self.fc = nn.Linear(self.bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        output = self.main(x)
        return output

# from UAN code
# adversarial network
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, max_train_step):
        super(AdversarialNetwork, self).__init__()
        self.mid_dim = in_feature // 2
        self.main = nn.Sequential(
            nn.Linear(in_feature, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=max_train_step))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y

# from UAN code
# adversarial network
class AdversarialNetwork2(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork2, self).__init__()
        self.mid_dim = in_feature // 2
        self.main = nn.Sequential(
            nn.Linear(in_feature, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, 1),
            nn.Sigmoid()
        )
        self.grl = GRL()

    def forward(self, x):
        x_ = self.grl.apply(x)
        y = self.main(x_)
        return y




class UDANLI(nn.Module):
    def __init__(self, model_name, num_class, max_train_step=0, **kwargs):
        super(UDANLI, self).__init__()
        print('INIT UDANLI...')
        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class

        try:
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.model.config.hidden_size
            
        self.classifier = Classifier(in_dim=self.hidden_dim, out_dim=2)
        # v1
        # self.discriminator = AdversarialNetwork(self.hidden_dim, max_train_step)
        # v2
        self.discriminator = AdversarialNetwork2(self.hidden_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        is_nli=True,
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

        # only for generating distribution per class
        if embeddings_only:
            return cls_state

        if is_nli:
            # shape : (batch, num_class)
            classification_output = self.classifier(cls_state)
            
            # shape : (batch, )
            predictions = classification_output.argmax(dim=-1)
            # shape : (batch, )
            max_logits = classification_output.max(dim=-1).values
            return  {
                'predictions' : predictions,
                'logits' : classification_output,
                'max_logits' : max_logits,
            }
        else:
            # shape : (batch, 1)
            domain_output = self.discriminator(cls_state)
            return  {
                'domain_output' : domain_output,
            }


        
