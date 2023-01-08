
import time
import logging
import copy
import os

import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pdb
# import ot as POT
import ot

# pdb.set_trace()

from torch import (
    nn,
    optim
)
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import (
    AutoTokenizer,
    get_scheduler,
)

from models.uniot import UniOT, MemoryQueue, sinkhorn, adaptive_filling, ubot_CCD
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)


# https://github.com/changwxx/UniOT-for-UniDA/blob/9ba3bad29956c2f170cd82c9dd8cfa3ae2af3dda/eval.py#L27
def eval(model, dataloader, gamma=0.7, beta=None):
    
    metric = Accuracy()

    
    model.eval()
    with torch.no_grad():
        label_list = []
        norm_feat_list = []
        for i, test_batch in tqdm(enumerate(dataloader), desc='Test Model'):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            label = test_batch['labels']

            # shape : (batch, hidden_size)
            feature = model(**test_batch)
            before_lincls_feat_t, after_lincls_t = model.classifier(feature)
            norm_feat_t = F.normalize(before_lincls_feat_t)
            
            label_list.append(label.cpu().data.numpy())
            norm_feat_list.append(norm_feat_t.cpu().data.numpy())

        # concatenate list
        label_list = np.concatenate(label_list)
        norm_feat_list = np.concatenate(norm_feat_list)

        # Unbalanced OT
        source_prototype = model.classifier.ProtoCLS.fc.weight

        stopThr = 1e-6
        # Adaptive filling 
        # norm_feat_list    : (batch, 256)
        # source_prototype  : (num_class, 256)
        newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_list).cuda(),
                                            source_prototype, gamma, beta, 0, stopThr=stopThr)

        # obtain predict label
        _, __, pred_label, ___ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr)
                    
        
        metric.add_batch(predictions=pred_label, references=torch.from_numpy(label_list).cuda())
    
    results = metric.compute()
    return results


# https://github.com/changwxx/UniOT-for-UniDA/blob/9ba3bad29956c2f170cd82c9dd8cfa3ae2af3dda/eval.py#L27
def test(model, dataloader, unknown_class, gamma=0.7, beta=None):
    metric = HScore(unknown_class)
    
    model.eval()
    with torch.no_grad():
        label_list = []
        norm_feat_list = []
        for i, test_batch in tqdm(enumerate(dataloader), desc='Test Model'):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            label = test_batch['labels']

            # shape : (batch, hidden_size)
            feature = model(**test_batch)
            before_lincls_feat_t, after_lincls_t = model.classifier(feature)
            norm_feat_t = F.normalize(before_lincls_feat_t)
            
            label_list.append(label.cpu().data.numpy())
            norm_feat_list.append(norm_feat_t.cpu().data.numpy())

        # concatenate list
        label_list = np.concatenate(label_list)
        norm_feat_list = np.concatenate(norm_feat_list)

        # Unbalanced OT
        source_prototype = model.classifier.ProtoCLS.fc.weight

        stopThr = 1e-6
        # Adaptive filling 
        # norm_feat_list    : (batch, 256)
        # source_prototype  : (num_class, 256)
        newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_list).cuda(),
                                            source_prototype, gamma, beta, 0, stopThr=stopThr)

        # obtain predict label
        _, __, pred_label, ___ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr)
                    
        
        metric.add_batch(predictions=pred_label, references=torch.from_numpy(label_list).cuda())
    
    results = metric.compute()
    return results

def main(args, save_config):
    seed_everything(args.train.seed)
    
    if args.dataset.num_source_class == args.dataset.num_common_class:
        is_cda = True
        split = 'cda'
    else:
        is_cda = False
        split = 'opda'

    # amazon reviews data
    if 'source_domain' in args.dataset:
        source_domain = args.dataset.source_domain
        target_domain = args.dataset.target_domain
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/ovanet/{source_domain}-{target_domain}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    # clinc, massive, trec
    else:
        source_domain = None
        target_domain = None
        coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
        ## LOGGINGS ##
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/uniot/{split}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    

    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    writer = SummaryWriter(log_dir)
    # dump configs
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##


    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')

    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATALOADER ##
    train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.train.batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain, drop_last=True)

    # pdb.set_trace()

    num_step_per_epoch = max(len(train_dataloader), len(train_unlabeled_dataloader))
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UniOT(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
        max_train_step=total_step,
        temp=args.train.temp,
        K = args.train.K,
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##


    # TODO : smaller lr for pre-trained model (?)
    # -> is this also valid in nlp?
    ## OPTIMIZER & SCHEDULER ##
    optimizer = optim.AdamW(model.parameters(), lr=args.train.lr)

    num_warmup_steps = int(total_step * 0.3)
    lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    ## OPTIMIZER & SCHEDULER ##

    # data iter
    source_iter = ForeverDataIterator(train_dataloader)
    target_iter = ForeverDataIterator(train_unlabeled_dataloader)

    # https://github.com/changwxx/UniOT-for-UniDA/blob/main/main.py#L63
    # Memory queue init
    target_size = train_dataloader.dataset.__len__()
    n_batch = int(args.train.MQ_size/args.train.batch_size)    
    memqueue = MemoryQueue(256, args.train.batch_size, n_batch, args.train.temp).cuda()
    cnt_i = 0
    with torch.no_grad():
        while cnt_i < n_batch:
            for i, batch in enumerate(train_dataloader):
                batch = {k: v.cuda() for k, v in batch.items()}
                id_target = batch['index']

                feature_ex = model(**batch)
                before_lincls_feat, after_lincls = model.classifier(feature_ex)
                memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
                cnt_i += 1
                if cnt_i > n_batch-1:
                    break

    # cross-entropy loss for classification
    ce = nn.CrossEntropyLoss().cuda()

    global_step = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0
    beta = None

    if args.train.train:
        logger.info(f'Start Training....')
        start_time = time.time()
        for current_epoch in range(1, args.train.num_train_epochs+1):
            model.train()

            # check early stop.
            if early_stop_count == args.train.early_stop:
                logger.info('Early stop. End.')
                break
            
            for current_step in tqdm(range(num_step_per_epoch), desc=f'TRAIN EPOCH {current_epoch}'):

                global_step += 1

                # optimizer zero-grad
                optimizer.zero_grad()

                ####################
                #                  #
                #     Load Data    #
                #                  #
                ####################

                ## source to cuda
                source_batch = next(source_iter)
                source_batch = {k: v.cuda() for k, v in source_batch.items()}
                source_labels = source_batch['labels']
                id_source = source_batch['index']

                ## source to cuda
                target_batch = next(target_iter)
                target_batch = {k: v.cuda() for k, v in target_batch.items()}
                id_target = target_batch['index']

                ####################
                #                  #
                #   Forward Pass   #
                #                  #
                ####################

                # Resnet
                # shape : (batch, 2048)
                feat_s = model(**source_batch)
                feat_t = model(**target_batch)

                # classifier
                # before_lincls_feat_s : (batch, 256)
                # after_lincls_t       : (batch, num_source_class)
                before_lincls_feat_s, after_lincls_s = model.classifier(feat_s)
                before_lincls_feat_t, after_lincls_t = model.classifier(feat_t)
                
                # normalize
                # shape : (batch, 256)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                # shape : (batch, K) = (batch, 50)
                after_cluhead_t = model.cluster_head(before_lincls_feat_t)

                ## Calculate CE loss
                loss_cls = ce(after_lincls_s, source_labels)

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

                # https://github.com/changwxx/UniOT-for-UniDA/blob/main/main.py#L149
                # =====Common Class Detection=====
                if global_step > num_step_per_epoch:
                    source_prototype = model.classifier.ProtoCLS.fc.weight
                    if beta is None:
                        beta = ot.unif(source_prototype.size()[0])

                    # fill input features with memory queue
                    fill_size_uot = n_batch * args.train.batch_size
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


                ####################
                #                  #
                #   Compute Loss   #
                #                  #
                ####################

                # total loss
                loss = loss_cls + args.train.lam * (loss_PCD + loss_CCD)

                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/loss_cls', loss_cls, global_step)
                writer.add_scalar('train/loss_PCD', loss_PCD, global_step)
                writer.add_scalar('train/loss_CCD', loss_CCD, global_step)

                # backward, optimization
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                        
                model.classifier.ProtoCLS.weight_norm() # very important for proto-classifier
                model.cluster_head.weight_norm() # very important for proto-classifier

                # if current_step == num_step_per_epoch-1:
                #     pdb.set_trace()

                # norm_feat_t : (batch, 256)
                # id_target   : (batch, )
                memqueue.update_queue(norm_feat_t, id_target.cuda())
                

            ####################
            #                  #
            #     Evaluate     #
            #                  #
            ####################
            
            if global_step > num_step_per_epoch:
                logger.info(f'Evaluate model at epoch {current_epoch} ...')

                # find optimal threshold from evaluation set (source domain) -> sub-optimal threshold
                results = eval(model, eval_dataloader, beta=beta)
                # write to tensorboard
                for k,v in results.items():
                    writer.add_scalar(f'eval/{k}', v, global_step)
                

                if results['accuracy'] > best_acc:
                    best_acc = results['accuracy']
                    best_results = results
                    early_stop_count = 0

                    print_dict(logger, string=f'\n* BEST TOTAL ACCURACY at epoch {current_epoch}', dict=results)

                    logger.info('Saving best model...')
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                    logger.info('Done saving...')
                else:
                    logger.info('\nNot best. Pass.')

                    early_stop_count += 1
                    logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')
            else:
                logger.info('Skip the first eval....')
        
        end_time = time.time()
        logger.info(f'Done training full step. Total time : {end_time-start_time}')

        # skip evaluation with low accuracy
        if best_results['accuracy'] < 55:
            logger.info(f'Low total accuracy {best_results["accuracy"]}. Skip testing.')
            exit() 

    else:
        logger.info('Skip training... ')
    
    ####################
    #                  #
    #       Test       #
    #                  #
    ####################

    logger.info('Loading best model ...')
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best.pth')))
            
    logger.info('Test model...')
    if is_cda:
        logger.info('TEST ON CDA SETTING.')    
        results = eval(model, test_dataloader)
        for k,v in results.items():
            writer.add_scalar(f'test/{k}', v, 0)

        print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

        logger.info('Done.')    
    else:
        logger.info('TEST WITH "UNKNOWN" CLASS.')
        results = test(model, test_dataloader, unknown_label, beta=beta)
        for k,v in results.items():
            writer.add_scalar(f'test/{k}', v, 0)

        print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

        logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

