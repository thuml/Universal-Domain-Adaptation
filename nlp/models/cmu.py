
from torch import nn

from transformers import AutoModel

from easydl import *

   

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
    def __init__(self, in_feature, max_train_step):
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
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=max_train_step))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y



class CMU(nn.Module):
    def __init__(self, model_name, num_class, max_train_step, **kwargs):
        super(CMU, self).__init__()
        print('INIT CMU ....')
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
        self.discriminator = AdversarialNetwork(256, max_train_step)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
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
        
        # cls_state         : (batch, hidden_dim)
        # after_bottlenack  : (batch, 256)
        # before_softmax    : (batch, num_class)
        # after_softmax     : (batch, num_class)
        cls_state, after_bottleneck, before_softmax, after_softmax = self.classifier(cls_state)
        