
import argparse
import time
import datetime
import logging

import ot
import easydict
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from tqdm import tqdm
from easydl import variable_to_numpy
from easydl import TrainingModeManager, Accumulator

from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import faiss
import pdb

from models.uniot import UniOT, MemoryQueue, sinkhorn, adaptive_filling, ubot_CCD
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

# https://github.com/changwxx/UniOT-for-UniDA/blob/9ba3bad29956c2f170cd82c9dd8cfa3ae2af3dda/eval.py#L27
def test(model, dataloader, unknown_class, gamma=0.7, beta=None, seed=None, uniformed_index=None):
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(dataloader):
            im = im.cuda()
            label = label.cuda()

            feature = model.feature_extractor(im)
            before_lincls_feat_t, after_lincls_t = model.classifier(feature)
            norm_feat_t = F.normalize(before_lincls_feat_t)
            

            # Unbalanced OT
            source_prototype = model.classifier.ProtoCLS.fc.weight

            stopThr = 1e-6
            # Adaptive filling 
            # newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_t).cuda(), 
            #                                     source_prototype, gamma, beta, 0, stopThr=stopThr)
            
            # norm_feat_t       : (batch, 256)
            # source_prototype  : (num_class, 256)
            newsim, fake_size = adaptive_filling(norm_feat_t.cuda(), 
                                                source_prototype, gamma, beta, 0, stopThr=stopThr)

            # obtain predict label
            _, __, pred_label, ___ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr)
                        
            metric.add_batch(predictions=pred_label, references=label)
    
    results = metric.compute()
    return results

def main(args, save_config):
    seed_everything(args.seed)
    
    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/uniot/{args.seed}/{args.train.lr}'
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
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes, return_id=True)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UniOT(args, source_classes).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    ## INIT MODEL ##


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
    optimizer_feature = optim.SGD(model.feature_extractor.parameters(), lr=args.train.lr * 0.1, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
    optimizer_cls = optim.SGD(model.classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
    optimizer_cluster = optim.SGD(model.cluster_head.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
    
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.train.min_step)
    opt_sche_feature = OptimWithSheduler(optimizer_feature, scheduler)
    opt_sche_cls = OptimWithSheduler(optimizer_cls, scheduler)
    opt_sche_cluster = OptimWithSheduler(optimizer_cluster, scheduler)
    # =================== optimizer

    # https://github.com/changwxx/UniOT-for-UniDA/blob/main/main.py#L63
    # Memory queue init
    target_size = target_train_dl.dataset.__len__()
    n_batch = int(args.train.MQ_size/args.data.dataloader.batch_size)    
    memqueue = MemoryQueue(256, args.data.dataloader.batch_size, n_batch, args.train.temp).cuda()
    cnt_i = 0
    with torch.no_grad():
        while cnt_i < n_batch:
            for i, (im_target, _, id_target) in enumerate(target_train_dl):
                im_target = im_target.cuda()
                id_target = id_target.cuda()
                feature_ex = model.feature_extractor(im_target)
                before_lincls_feat, after_lincls = model.classifier(feature_ex)
                memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
                cnt_i += 1
                if cnt_i > n_batch-1:
                    break

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

    # CE-loss
    ce = nn.CrossEntropyLoss().cuda()
    
    
    current_epoch = 0
    best_hscore = 0
    best_results = None
    early_stop_count = 0
    beta = None

    ## START TRAINING ##
    for global_step in tqdm(range(args.train.min_step), desc='Train Model'):

        model.train()

        ####################
        #                  #
        #   Forward Pass   #
        #                  #
        ####################

        # image, label, id 
        # im_source     : (batch, w, h)
        # label_source  : (batch, )
        im_source, label_source, _ = next(source_iter)
        im_target, _, _  = next(target_iter)

        # pdb.set_trace()


        label_source = label_source.cuda()
        im_source = im_source.cuda()
        im_target = im_target.cuda()


        feat_s = model.feature_extractor(im_source)
        feat_t = model.feature_extractor(im_target)


        before_lincls_feat_s, after_lincls_s = model.classifier(feat_s)
        before_lincls_feat_t, after_lincls_t = model.classifier(feat_t)

        norm_feat_s = F.normalize(before_lincls_feat_s)
        norm_feat_t = F.normalize(before_lincls_feat_t)

        after_cluhead_t = model.cluster_head(before_lincls_feat_t)



        # =====Private Class Discovery=====
        minibatch_size = norm_feat_t.size(0)

        # obtain nearest neighbor from memory queue and current mini-batch
        feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / args.train.temp
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1 / args.train.temp)

        nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target.cuda())
        neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
        values, indices = torch.max(neighbor_candidate_sim, 1)
        neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).cuda()
        for i in range(minibatch_size):
            neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
            neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
            
        neighbor_output = model.cluster_head(neighbor_norm_feat)
        
        # fill input features with memory queue
        fill_size_ot = args.train.K
        mqfill_feat_t = memqueue.random_sample(fill_size_ot)
        mqfill_output_t = model.cluster_head(mqfill_feat_t)

        # OT process
        # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
        S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        S_tt *= args.train.temp
        Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
        Q_tt_tilde = Q_tt * Q_tt.size(0)
        anchor_Q = Q_tt_tilde[:minibatch_size, :]
        neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]


        # https://github.com/changwxx/UniOT-for-UniDA/blob/main/main.py#L149
        # =====Common Class Detection=====
        if global_step > 100:
            source_prototype = model.classifier.ProtoCLS.fc.weight
            if beta is None:
                beta = ot.unif(source_prototype.size()[0])

            # fill input features with memory queue
            fill_size_uot = n_batch * args.data.dataloader.batch_size
            mqfill_feat_t = memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            full_size = ubot_feature_t.size(0)
            
            # Adaptive filling
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, args.train.gamma, beta, fill_size_uot)
        
            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, beta, fake_size=fake_size, 
                                                                    fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            beta = args.train.mu*beta + (1-args.train.mu)*new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            # Due to mini-batch sampling, current mini-batch samples might be all target-private. 
            # (especially when target-private samples dominate target domain, e.g. OfficeHome)
            if high_conf_label_id.size(0) > 0:
                loss_CCD = ce(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0
        else:
            loss_CCD = 0
        
        # =====Source Supervision=====      
        loss_cls = ce(after_lincls_s, label_source)

        # compute loss_PCD
        loss_local = 0
        for i in range(minibatch_size):
            sub_loss_local = 0
            sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
            sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
            sub_loss_local /= 2
            loss_local += sub_loss_local
        loss_local /= minibatch_size
        loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
        loss_PCD = (loss_global + loss_local) / 2
        

        # total loss
        loss_all = loss_cls + args.train.lam * (loss_PCD + loss_CCD)


        with OptimizerManager([opt_sche_feature, opt_sche_cls, opt_sche_cluster]):
            loss_all.backward()


        global_step += 1

        model.classifier.ProtoCLS.weight_norm() # very important for proto-classifier
        model.cluster_head.weight_norm() # very important for proto-classifier
        memqueue.update_queue(norm_feat_t, id_target.cuda())

        ####################
        #                  #
        #     Logging      #
        #                  #
        ####################

        if global_step % log_interval == 0:
            writer.add_scalar('train/loss_cls', loss_cls, current_epoch)
            writer.add_scalar('train/loss_PCS', loss_PCD, current_epoch)
            writer.add_scalar('train/loss_CCD', loss_CCD, current_epoch)
            writer.add_scalar('train/loss', loss_all, current_epoch)


        ####################
        #                  #
        #       Test       #
        #                  #
        ####################
        
        if global_step % test_interval == 0:
            current_epoch += 1
            logger.info(f'TEST at epoch {current_epoch} ...')
            results = test(model, target_test_dl, unknown_class, beta=beta)
            writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], global_step)
            writer.add_scalar('test/total_acc_test', results['total_accuracy'], global_step)
            writer.add_scalar('test/known_test', results['known_accuracy'], global_step)
            writer.add_scalar('test/unknown_test', results['unknown_accuracy'], global_step)
            writer.add_scalar('test/hscore_test', results['h_score'], global_step)


            if results['h_score'] > best_hscore:
                best_hscore = results['h_score']
                best_results = results
                early_stop_count = 0

                print_dict(logger, string=f'* Best H-score at epoch {current_epoch}', dict=results)

                logger.info('Saving best model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                logger.info('Done saving...')
            else:
                print_dict(logger, string=f'* Current Results at epoch {current_epoch}', dict=results)

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

    
    print_dict(logger, string=f'** BEST RESULTS', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

