
import torch
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
        self.discriminator = AdversarialNetwork(self.hidden_dim, max_train_step)

    # https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L226
    def norm(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        return x

    # https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L179
    def get_consistency(self, y_1, y_2, y_3, y_4, y_5):
        y_1 = torch.unsqueeze(y_1, 1)
        y_2 = torch.unsqueeze(y_2, 1)
        y_3 = torch.unsqueeze(y_3, 1)
        y_4 = torch.unsqueeze(y_4, 1)
        y_5 = torch.unsqueeze(y_5, 1)
        c = torch.cat((y_1, y_2, y_3, y_4, y_5), dim=1)
        d = torch.std(c, 1)
        consistency = torch.mean(d, 1)
        return consistency

    # https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L191
    def get_entropy(self, y_1, y_2, y_3, y_4, y_5):
        y_1 = nn.Softmax(-1)(y_1)
        y_2 = nn.Softmax(-1)(y_2)
        y_3 = nn.Softmax(-1)(y_3)
        y_4 = nn.Softmax(-1)(y_4)
        y_5 = nn.Softmax(-1)(y_5)

        entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
        entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
        entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
        entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
        entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
        entropy_norm = np.log(y_1.size(1))

        entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
        return entropy

    # https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L203
    def single_entropy(self, y_1):
        entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
        entropy_norm = np.log(y_1.size(1))
        entropy = entropy1 / entropy_norm
        return entropy

    # https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L210
    def get_confidence(self, y_1, y_2, y_3, y_4, y_5):
        conf_1, indice_1 = torch.max(y_1, 1)
        conf_2, indice_2 = torch.max(y_2, 1)
        conf_3, indice_3 = torch.max(y_3, 1)
        conf_4, indice_4 = torch.max(y_4, 1)
        conf_5, indice_5 = torch.max(y_5, 1)
        confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
        return confidence

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
        # feature           : (batch, 256)
        # fc2_s ~ fc2_s5    : (batch, num_class)
        # predict_prob      : (batch, num_class)
        cls_state, feature, fc2_1, fc2_2, fc2_3, fc2_4, fc2_5, predict_prob = self.classifier(cls_state)
        
        # shape : (batch, )
        predictions = predict_prob.argmax(dim=-1)

        entropy = self.get_entropy(fc2_1, fc2_2, fc2_3, fc2_4, fc2_5).detach()
        consistency = self.get_consistency(fc2_1, fc2_2, fc2_3, fc2_4, fc2_5).detach()
        confidence = self.get_confidence(fc2_1, fc2_2, fc2_3, fc2_4, fc2_5).detach()

        entropy = self.norm(torch.tensor(entropy))
        consistency = self.norm(torch.tensor(consistency))
        confidence = self.norm(torch.tensor(confidence))

        # TODO : re?
        weight = (1 - entropy + 1 - consistency + confidence) / 3           

    
        # shape             : (batch, 1)
        domain_output = self.discriminator(cls_state)


        # shape : (batch, )
        max_logits = predict_prob.max(dim=-1).values

        return {
            'predictions' : predictions,
            'logits' : predict_prob,
            'fc2_1' : fc2_1,
            'fc2_2' : fc2_2,
            'fc2_3' : fc2_3,
            'fc2_4' : fc2_4,
            'fc2_5' : fc2_5,
            # 'max_logits' : weight,
            'max_logits' : max_logits,
            'domain_output' : domain_output,
        }