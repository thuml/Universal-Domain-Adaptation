
import ot
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import models

from easydl import *

from easydl import AccuracyCounter
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

class ResultsCalculator(object):
    """
    calculate final results (including overall acc, common acc, target-private(tp) acc, h-score, tp NMI, h3-score)
    """
    def __init__(self, classes_set, label, predict_label, private_label, private_pred, uniformed_index=None):
        if uniformed_index is None:
            uniformed_index = max(classes_set['source_classes']) + 1
        self.overall_accs, self.overall_acc_aver = self._get_overall_acc(classes_set['source_classes'], label, predict_label, uniformed_index)
        common_accs = dict()
        for index in classes_set['common_classes']:
            common_accs[index] = self.overall_accs[index]
        self.common_acc_aver = np.mean(list(common_accs.values()))
        self.tp_acc = self.overall_accs[uniformed_index]
        self.h_score = 2*self.common_acc_aver*self.tp_acc / (self.common_acc_aver+self.tp_acc)
        self.tp_nmi = self._nmi_cal(private_label, private_pred)
        self.h3_score = 3*(1/ (1/self.common_acc_aver + 1/self.tp_acc + 1/self.tp_nmi))
        
    def _get_overall_acc(self,source_classes, label, predict_label, uniformed_index):
        unif_label = self._tplabel_unif(source_classes, label, uniformed_index)
        overall_accs, overall_acc_aver = self._recall_acc_cal(source_classes+[uniformed_index], predict_label, unif_label)
        return overall_accs, overall_acc_aver

    def _recall_acc_cal(self, classes, predict_label, label):
        counters = {class_label:AccuracyCounter() for class_label in classes}
        for (each_predict_label, each_label) in zip(predict_label, label):
            if each_label in classes:
                counters[each_label].Ntotal += 1.0
                if each_predict_label == each_label:
                    counters[each_label].Ncorrect += 1.0
        recall_accs = {i:counters[i].reportAccuracy() for i in counters.keys() \
                                                        if not np.isnan(counters[i].reportAccuracy())}
        recall_acc_aver = np.mean(list(recall_accs.values()))
        return recall_accs, recall_acc_aver
    
    def _tplabel_unif(self, source_classes, label, uniformed_index):
        uniform_tp = label.copy()
        for i in range(len(label)):
            if label[i] not in source_classes:
                uniform_tp[i] = uniformed_index
        return uniform_tp

    def _nmi_cal(self, label, proto_pred):
        nmi = normalized_mutual_info_score(label, proto_pred)
        return nmi

    def _hungarian_matching(self, label, proto_pred):
        assert proto_pred.size == label.size
        matrix_size = max(len(set(proto_pred)), len(set(label)))
        matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
        for i in range(proto_pred.size):
            matrix[proto_pred[i], label[i]] += 1
        mat_index = linear_assignment(matrix.max() - matrix)
        return matrix,mat_index
    
    def _clutser_acc_cal(self, common_classes, tp_classes, label, proto_pred):
        classes = []
        classes.extend(common_classes)
        classes.extend(tp_classes)
        counters = {class_label:AccuracyCounter() for class_label in classes}
        for index in label:
            counters[index].Ntotal += 1
        index2label_map = {i:classes[i] for i in range(len(classes))}
        label2index_map = {classes[i]:i for i in range(len(classes))}
        transform_label = np.zeros_like(label)
        for i in range(len(label)):
            transform_label[i] = label2index_map[label[i]]
        matrix,mat_index = self._hungarian_matching(transform_label, proto_pred)
        sum_Ncorrect = 0
        for (i,j) in zip(mat_index[0],mat_index[1]):
            if index2label_map.get(j) is not None:
                counters[index2label_map[j]].Ncorrect = matrix[i,j]
                sum_Ncorrect += matrix[i,j]
        
        self.traditional_acc = sum_Ncorrect * 1.0 / len(proto_pred)

        self.overall_accs = {i:counters[i].reportAccuracy() for i in counters.keys() \
                                                        if not np.isnan(counters[i].reportAccuracy())}
        self.overall_acc = np.mean(list(self.overall_accs.values()))
        
        common_accs = dict()
        for index in common_classes:
            common_accs[index] = self.overall_accs[index]
        self.common_acc = np.mean(list(common_accs.values()))

        tp_accs = dict()
        for index in tp_classes:
            if index in self.overall_accs:
                tp_accs[index] = self.overall_accs[index]
        self.tp_acc = np.mean(list(tp_accs.values()))


class BaseFeatureExtractor(nn.Module):
    '''
    From https://github.com/thuml/Universal-Domain-Adaptation
    a base class for feature extractor
    '''
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    modefied from https://github.com/thuml/Universal-Domain-Adaptation
    implement ResNet50 as backbone, but the last fc layer is removed
    ** input image should be in range of [0, 1]**
    """
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        # init resnet
        model_resnet = models.resnet50(pretrained=True)
        # model_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        
        # model_resnet = models.resnet50(pretrained=False)
        # model_resnet.load_state_dict(torch.load('/home/heyjoonkim/data/resnet50.pth'))

        # pretrain model is used, use ImageNet normalization
        self.normalize = True
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        mod = list(model_resnet.children())
        mod.pop()
        self.feature_extractor = nn.Sequential(*mod)
        self.output_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x


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
    def __init__(self, args, source_classes, **kwargs):
        super(UniOT, self).__init__()
        print('INIT UniOT...')
        self.num_class = len(source_classes)
        self.hidden_dim = 2048
        self.temp = args.train.temp
        self.K = args.train.K
        self.feature_dim = 256
    
        self.feature_extractor = ResNet50Fc()
        self.classifier = CLS(self.feature_extractor.output_dim, self.num_class, hidden_mlp=2048, feat_dim=self.feature_dim, temp=self.temp) 
        self.cluster_head = ProtoCLS(in_dim=self.feature_dim, out_dim=self.K, temp=self.temp)
    
    def train_base_model(self, mode):
        self.feature_extractor.train(mode)

    def get_feature(self, x):
        
        # shape : (batch, hidden_dim)
        feat = self.feature_extractor(x)

        return feat
    
    def get_prediction_and_logits(self, x):
        # feature               : (batch, hidden_dim)
        # before_lincls_feat_t  : (batch, 256)
        # predict_prob          : (batch, num_source_label)
        feature = self.feature_extractor(x)
        before_lincls_feat_t, predict_prob = self.classifier(feature)

        # shape : (batch, )
        predictions = predict_prob.argmax(dim=-1)
        max_logits = predict_prob.max(dim=-1).values

        return {
            'predictions' : predictions,
            'total_logits' : predict_prob,
            'max_logits' : max_logits
        }


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











