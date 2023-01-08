
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from transformers import AutoModel

from easydl import *

import ot

class ProtoCLS(nn.Module):
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls

class UniOT(nn.Module):
    def __init__(self, model_name, num_class, temp, K, **kwargs):
        super(UniOT, self).__init__()
        print('INIT UniOT...')
        
        self.model_name = model_name
        self.num_class = num_class
        self.unk_index = num_class

        self.temp = temp
        self.K = K
        self.feature_dim = 256

        try:
            self.feature_extractor = AutoModel.from_pretrained(self.model_name)
        except:
            print(f'Unable to load model {self.model_name}')
            exit()

        self.hidden_dim = self.feature_extractor.config.hidden_size

        self.classifier = CLS(self.hidden_dim, self.num_class, hidden_mlp=2048, feat_dim=self.feature_dim, temp=self.temp) 
        self.cluster_head = ProtoCLS(in_dim=self.feature_dim, out_dim=self.K, temp=self.temp)
    


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs,
    ):

        outputs = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        batch_size, input_length = input_ids.shape

        # shape : (batch, length, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        # shape : (batch, hidden_dim)
        cls_state = last_hidden_state[:, 0, :]

        return cls_state


class MemoryQueue(nn.Module):
    def __init__(self, feat_dim, batchsize, n_batch, T=0.05):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.batchsize = batchsize
        self.T = T

        # init memory queue
        self.queue_size = self.batchsize * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros((self.queue_size), dtype=int))
        self.mem_feat = self.mem_feat.cuda()
        self.mem_id = self.mem_id.cuda()

        # write pointer
        self.next_write = 0

    def forward(self, x):
        """
        obtain similarity between x and the features stored in memory queue指针 英语
        """
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        """
        get anchors' nearest neighbor in memory queue 
        """
        # compute similarity first
        feat_mat = self.forward(anchors)

        # assign the similarity between features of the same sample with -1/T
        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:,1]
            feat_mat[:, temp] = -1 / self.T

        # obtain neighbor's similarity value and corresponding feature
        values, indices = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).cuda()
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i],:]
        return values, nearest_feat

    def update_queue(self, features, ids):
        """
        update memory queue
        """
        w_ids = torch.arange(self.next_write, self.next_write+self.batchsize).cuda()
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)

        # update write pointer
        self.next_write += self.batchsize
        if self.next_write == self.queue_size:
            self.next_write = 0

    def random_sample(self, size):
        """
        sample some features from memory queue randomly
        """ 
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long().cuda()
        sample_feat = self.mem_feat[id_t]
        return sample_feat

# https://github.com/changwxx/UniOT-for-UniDA/blob/main/utils/lib.py#L16
def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

# https://github.com/changwxx/UniOT-for-UniDA/blob/main/utils/lib.py#L42
def ubot_CCD(sim, beta, fake_size=0, fill_size=0, mode='minibatch', stopThr=1e-4):
    # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
    M = -sim                         
    alpha = ot.unif(sim.size(0))
    
    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta, M.detach().cpu().numpy(), 
                                                    reg=0.01, reg_m=0.5, stopThr=stopThr) 
    Q_st = torch.from_numpy(Q_st).float().cuda()

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st/sum_pi
    
    # highly confident target samples selected by statistics mean
    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size+fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar

    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    # confidence score w^s_j
    ws_j = torch.sum(Q_st_bar, 0)

    # filter by statistics mean
    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1/Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1/Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)
    
    # for adaptive update
    new_beta = torch.sum(Q_st_bar,0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta

# https://github.com/changwxx/UniOT-for-UniDA/blob/main/utils/lib.py#L80
def adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size, stopThr=1e-4):
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim,1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0)/max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        # positive filling
        fake_size = neg_num - pos_num
        if fake_size > 0:
            # do 1st OT find top confident target samples
            high_conf_label_id, _, __, ___ = ubot_CCD(sim, beta, fake_size=0, fill_size=fill_size, 
                                                    mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).cuda()
                fill_pos = sim[high_conf_label_id[select_id],:] 
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        # negative filling
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id,:]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).cuda()
            fill_neg = fake_private[select_id,:]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim
    
    return newsim, fake_size