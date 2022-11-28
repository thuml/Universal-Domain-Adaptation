
import argparse
import time
import datetime
import logging

import easydict
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Function
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.dance import DANCE
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything
from utils.evaluation import HScore
from utils.data import *

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--lr', type=float, default=None, help='Custom learning rate.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')

    args = parser.parse_args()
    lr = args.lr
    seed = args.seed

    config_file = args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    if lr is not None:
        args.train.lr = lr
    args.seed = seed

    return args, save_config

# https://github.com/VisionLearningGroup/DANCE/blob/6d84da24961aca75b011ed5fe185f36c4a3c1b88/models/LinearAverage.py#L6
class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]));
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory =  self.memory.cuda()

    def forward(self, x, y):
        # x             : (batch, hidden_dim)
        # self.memory   : (num_samples, hidden_dim)
        # out           : (batch, num_samples)
        out = torch.mm(x, self.memory.t())/self.T
        return out

    def update_weight(self, features, index):
        weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
        weight_pos.mul_(0.0)
        weight_pos.add_(torch.mul(features.data, 1.0))

        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, index, updated_weight)
        self.flag = 1
        self.memory = F.normalize(self.memory)#.cuda()


    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)

# https://github.com/VisionLearningGroup/DANCE/blob/6d84da24961aca75b011ed5fe185f36c4a3c1b88/utils/loss.py#L4
def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

# https://github.com/VisionLearningGroup/DANCE/blob/6d84da24961aca75b011ed5fe185f36c4a3c1b88/utils/loss.py#L12
def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)

# https://github.com/VisionLearningGroup/DANCE/blob/6d84da24961aca75b011ed5fe185f36c4a3c1b88/utils/loss.py#L8
def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))

# https://github.com/VisionLearningGroup/DANCE/blob/6d84da24961aca75b011ed5fe185f36c4a3c1b88/utils/lr_schedule.py#L2
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001,weight_decay=0.0005, max_iter=10000):
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return 

def test(dataloader, model, source_classes, unknown_class, threshold):
    metric = HScore(unknown_class)

    model.G.eval()
    model.C.eval()
   
    with torch.no_grad():
        for i, (im, labels) in enumerate(dataloader):
            im = im.cuda()
            labels = labels.cuda()

            feat = model.G(im)
            out = model.C(feat)
            # shape : (batch, num_source_class)
            out = F.softmax(out)
            # shape : (batch, )
            entr = -torch.sum(out * torch.log(out), 1).data
            # shape : (batch, )
            predictions = out.max(1).indices
            predictions = predictions
            unk = entr > threshold
            predictions[unk] = unknown_class

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    return results

    
def main(args, save_config):
    seed_everything(args.seed)
    
    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/dance/{args.seed}/{args.train.lr}'
    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    if not args.test.test_only:
        writer = SummaryWriter(log_dir)
    # dump configs
    with open(join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##


    ## LOAD DATASETS ##
    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = DANCE(args, source_classes).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    ## INIT MODEL ##

    lemniscate = LinearAverage(inputSize=2048, outputSize=len(target_train_dl.dataset), T=0.005, momentum=args.train.momentum).cuda()

    ## TEST ONLY ##
    if args.test.test_only:
        logger.info('TEST ONLY...')
        state_dict_path = os.path.join(log_dir, 'best.pth')
        assert os.path.exists(state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))
        results = test(model, target_test_dl, unknown_class)

        print_dict(logger, string='======== Final Test Results ========', dict=results)
        exit(0)
    ## TEST ONLY ##

    # =================== optimizer    
    params = []
    for key, value in dict(model.G.named_parameters()).items():
        if value.requires_grad and "features" in key:
            if 'bias' in key:
                params += [{'params': [value], 'lr': args.train.multi,
                            'weight_decay': args.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': args.train.multi,
                            'weight_decay': args.train.weight_decay}]
        else:
            if 'bias' in key:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': args.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': args.train.weight_decay}]   
                
    opt_g = optim.SGD(params, momentum=args.train.sgd_momentum,
                  weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(model.C.parameters()), lr=1.0,
                   momentum=args.train.sgd_momentum, weight_decay=args.train.temp,
                   nesterov=True)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in opt_c.param_groups:
        param_lr_f.append(param_group["lr"])
    # =================== optimizer


    # total steps / epochs
    steps_per_epoch = max(len(source_train_dl), len(target_train_dl))
    total_epoch = round(args.train.min_step / steps_per_epoch)
    logger.info(f'Total epoch {total_epoch}, steps per epoch {steps_per_epoch}, total step {args.train.min_step}')

    # log every epoch
    log_interval = steps_per_epoch
    # test every epoh
    test_interval = steps_per_epoch

    logger.info(f'Start Training....')
    start_time = time.time()

    source_iter = ForeverDataIterator(source_train_dl)
    target_iter = ForeverDataIterator(source_train_dl)
    num_target_iter = len(target_iter)

    # CE-loss
    ce = nn.CrossEntropyLoss().cuda()

    logger.info('Start main training...')

    current_epoch = 0
    best_acc = 0
    best_hscore = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
    for current_step in tqdm(range(args.train.min_step), desc='Train Model'):        
        ####################
        #                  #
        #       Train      #
        #                  #
        ####################

        model.G.train()
        model.C.train()

        # scheduler
        inv_lr_scheduler(param_lr_g, opt_g, current_step,
                         init_lr=args.train.lr,
                         max_iter=args.train.min_step)
        inv_lr_scheduler(param_lr_f, opt_c, current_step,
                         init_lr=args.train.lr,
                         max_iter=args.train.min_step)
        
        # optimizer
        opt_g.zero_grad()
        opt_c.zero_grad()
        # normalize
        model.C.weight_norm()

        ## source domain 
        im_s, label_s = next(source_iter)
        im_s = im_s.cuda()
        label_s = label_s.cuda()
        
        feat_s = model.G(im_s)
        out_s = model.C(feat_s)

        # classification loss
        loss_s = ce(out_s, label_s)

        ## target domain
        im_t, label_t = next(target_iter)
        im_t = im_t.cuda()

        # shape : (batch, hidden_dim)
        feat_t = model.G(im_t)
        # shape : (batch, num_source_class)
        out_t = model.C(feat_t)
        # shape : (batch, hidden_dim)
        feat_t = F.normalize(feat_t)

        
        # calculate target index
        batch_index = current_step % num_target_iter 
        batch_size = args.data.dataloader.batch_size
        current_batch_size, _ = out_t.shape
        start_index = batch_index * batch_size
        end_index = start_index + current_batch_size
        # shape : (batch_size, )
        index_t = torch.arange(start_index, end_index).cuda()

        print(out_t.shape, batch_index, batch_size, current_batch_size, start_index, end_index)
        
        # if batch_index == 10:
        #     pdb.set_trace()

        # calculate mini-batch x memory similarity
        # shape : (batch, num_samples)
        feat_mat = lemniscate(feat_t, index_t)

        
        # if batch_index == 10:
        #     pdb.set_trace()

        feat_mat[:, index_t] = -1 / args.train.temp

        ### Calculate mini-batch x mini-batch similarity
        # shape : (batch, batch)
        feat_mat2 = torch.matmul(feat_t, feat_t.t()) / args.train.temp
        # shape : (batch, batch)
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1 / args.train.temp)
        loss_nc = args.train.eta * entropy(torch.cat([out_t, feat_mat,
                                                      feat_mat2], 1))
        loss_ent = args.train.eta * entropy_margin(out_t, args.train.thr,
                                                   args.train.margin)
        
        # total loss
        loss = loss_nc + loss_s + loss_ent

        # update
        loss.backward()
        opt_g.step()
        opt_c.step()
        lemniscate.update_weight(feat_t, index_t)
        
        ####################
        #                  #
        #     Logging      #
        #                  #
        ####################
        writer.add_scalar('train/nc_loss', loss_nc, current_step)
        writer.add_scalar('train/source_loss', loss_s, current_step)
        writer.add_scalar('train/ent_loss', loss_ent, current_step)
        writer.add_scalar('train/loss', loss, current_step)


        ####################
        #                  #
        #       Test       #
        #                  #
        ####################
        current_step += 1
        if current_step % test_interval == 0:
            logger.info(f'TEST at epoch {current_epoch} ...')
            results = test(target_test_dl, model, source_classes, unknown_class, args.train.thr)
            writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], current_epoch)
            writer.add_scalar('test/total_acc_test', results['total_accuracy'], current_epoch)
            writer.add_scalar('test/known_test', results['known_accuracy'], current_epoch)
            writer.add_scalar('test/unknown_test', results['unknown_accuracy'], current_epoch)
            writer.add_scalar('test/hscore_test', results['h_score'], current_epoch)


            if results['h_score'] > best_hscore:
                best_hscore = results['h_score']
                best_results = results
                early_stop_count = 0

                # print_dict(logger, string=f'* Best accuracy at epoch {current_epoch}', dict=results)
                print_dict(logger, string=f'* Best h-score at epoch {current_epoch}', dict=results)

                logger.info('Saving best model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                logger.info('Done saving...')
            else:
                print_dict(logger, string=f'* Results at epoch {current_epoch}', dict=results)

                logger.info('Saving current model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'current.pth'))
                logger.info('Done saving...')

                if early_stop_count == args.train.early_stop:
                    logger.info('End.')
                    end_time = time.time()
                    logger.info(f'Done training at epoch {current_epoch}. Total time : {end_time-start_time}')     

                    print_dict(logger, string=f'** BEST RESULTS', dict=best_results)

                    exit()
                early_stop_count += 1
                logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')
            current_epoch += 1

    
    print_dict(logger, string=f'** BEST RESULTS', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

