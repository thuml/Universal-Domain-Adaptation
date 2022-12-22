
from torch import nn
from torchvision import models

from transformers import AutoModel

from easydl import *

# from UAN code
# classification head
class Classifier(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(Classifier, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        output = self.main(x)
        return output

# from UAN code
# adversarial network
class AdversarialNetwork(nn.Module):
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




class DANN(nn.Module):
    def __init__(self, model_name, num_class, max_train_step, **kwargs):
        super(DANN, self).__init__()
        print('INIT DANN...')
        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class

        try:
            self.model = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.model.config.hidden_size
            
        self.classifier = Classifier(in_dim=self.hidden_dim, out_dim=self.num_class, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(self.hidden_dim, max_train_step)

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

        # shape : (batch, num_class)
        classification_output = self.classifier(cls_state)
        # shape : (batch, 1)
        domain_output = self.discriminator(cls_state)

        # shape : (batch, )
        predictions = classification_output.argmax(dim=-1)
        # shape : (batch, )
        max_logits = classification_output.max(dim=-1).values

        return  {
            'predictions' : predictions,
            'logits' : classification_output,
            'max_logits' : max_logits,
            'domain_output' : domain_output,
        }
