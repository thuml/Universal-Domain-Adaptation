
import argparse
import time
import datetime
import logging
import copy

import easydict
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from torch import nn
from tqdm import tqdm

from torch import optim
from tensorboardX import SummaryWriter
import pdb

from models.dann import DANN
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
    parser.add_argument('--min_threshold', type=float, default=0.0, help='Minimum threshold value.')
    parser.add_argument('--max_threshold', type=float, default=1.0, help='Maximum threshold value.')
    parser.add_argument('--step', type=float, default=0.005, help='Step value.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')


    args = parser.parse_args()
    lr = args.lr
    min_threshold = args.min_threshold
    max_threshold = args.max_threshold
    step = args.step
    seed = args.seed


    config_file = args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    if lr is not None:
        args.train.lr = lr

    args.min_threshold = min_threshold
    args.max_threshold = max_threshold
    args.step = step
    args.seed = seed

    return args, save_config

def test(model, dataloader, unknown_class):
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(dataloader):
            im = im.cuda()
            label = label.cuda()

            # predictions   : (batch, )
            # max_logits    : (batch, )
            # total_logits  : (batch, num_source_class)
            outputs  = model.get_prediction_and_logits(im)
            predictions, _, _ = outputs['predictions'], outputs['total_logits'], outputs['max_logits']
            
            metric.add_batch(predictions=predictions, references=label)
    
    results = metric.compute()
    return results


def cheating_test(model, dataloader, unknown_class, start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(dataloader, desc='testing ')):
            im = im.cuda()
            label = label.cuda()

            # predictions   : (batch, )
            # max_logits    : (batch, )
            # total_logits  : (batch, num_source_class)
            outputs  = model.get_prediction_and_logits(im)
            predictions, total_logits, max_logits = outputs['predictions'], outputs['total_logits'], outputs['max_logits']

            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                unknown = (max_logits < threshold).squeeze()
                tmp_predictions[unknown] = unknown_class

                metrics[index].add_batch(
                    predictions=tmp_predictions,
                    references=label
                )
    best_threshold = 0
    best_accuracy = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = metrics[index].compute()
        current_accuracy = results['mean_accuracy'] * 100

        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            best_threshold = threshold
            best_results = results

    return best_results, best_threshold

# from original code
# https://github.com/VisionLearningGroup/OVANet/blob/d40020d2d59e617ca693ce5195b7b5a44a9893d5/utils/lr_schedule.py#L2
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10,
                     power=0.75, init_lr=0.001,weight_decay=0.0005,
                     max_iter=10000):
    #10000
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #max_iter = 10000
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return lr

def main(args, save_config):
    seed_everything(args.seed)

    ## GPU SETTINGS ##
    # gpu_ids = select_GPUs(args.misc.gpus)
    # TODO : remove?
    gpu_ids = [0]
    output_device = gpu_ids[0]
    ## GPU SETTINGS ##

    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/dann/{args.seed}/{args.train.lr}'
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
    model = DANN(args, source_classes).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')


    # model = nn.DataParallel(model, device_ids=gpu_ids, output_device=output_device)
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
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_base = OptimWithSheduler(
        optim.SGD(model.base_model.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(list(model.classifier.parameters()) + list(model.domain_discriminator.parameters()), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)

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

    # CE-loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    # BCE-loss for domain classification
    bce = nn.BCELoss().cuda()
    
    current_epoch = 0
    best_acc = 0
    best_threshold = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
    for global_step in tqdm(range(args.train.min_step), desc='Train Model'):
        global_step += 1
        
        model.train()

        ####################
        #                  #
        #   Forward Pass   #
        #                  #
        ####################

        # get samples
        im_source, label_source = next(source_iter)
        im_target, _  = next(target_iter)

        # to cuda
        label_source = label_source.cuda()
        im_source = im_source.cuda()
        im_target = im_target.cuda()

        # optimizer zero grad
        optimizer_base.zero_grad()
        optimizer_cls.zero_grad()

        ## adversarial training : source = 1, target = 0

        ## source
        # classification_s  : (batch, num_source_class)
        # domain_s          : (batch, 1)
        classification_s, domain_s = model(im_source)

        ## target
        # domain_t          : (batch, 1)
        _, domain_t = model(im_target)

                
        ####################
        #                  #
        #   Compute Loss   #
        #                  #
        ####################

        # source bce loss
        bce_loss_s = bce(domain_s, torch.ones_like(domain_s))
        # source ce loss
        ce_loss_s = ce(classification_s, label_source)
        
        # target bce_loss
        bce_loss_t = bce(domain_t, torch.zeros_like(domain_t))

        # total loss
        loss = ce_loss_s + bce_loss_s + bce_loss_t

        # backward + step
        loss.backward()
        optimizer_base.step()
        optimizer_cls.step()


        ####################
        #                  #
        #     Logging      #
        #                  #
        ####################

        if global_step % log_interval == 0:
            writer.add_scalar('train/ce_loss', ce_loss_s, current_epoch)
            writer.add_scalar('train/adv_loss_source', bce_loss_s, current_epoch)
            writer.add_scalar('train/adv_loss_target', bce_loss_t, current_epoch)
            writer.add_scalar('train/loss', loss, current_epoch)


        ####################
        #                  #
        #       Test       #
        #                  #
        ####################
        
        if global_step % test_interval == 0:
            current_epoch += 1
            logger.info(f'TEST at epoch {current_epoch} ...')
            results, threshold = cheating_test(model, target_test_dl, unknown_class, start=args.min_threshold, end=args.max_threshold, step=args.step)    
            writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], global_step)
            writer.add_scalar('test/total_acc_test', results['total_accuracy'], global_step)
            writer.add_scalar('test/known_test', results['known_accuracy'], global_step)
            writer.add_scalar('test/unknown_test', results['unknown_accuracy'], global_step)
            writer.add_scalar('test/hscore_test', results['h_score'], global_step)
            writer.add_scalar('test/threshold', threshold, global_step)


            if results['mean_accuracy'] > best_acc:
                best_acc = results['mean_accuracy']
                best_results = results
                best_threshold = threshold
                early_stop_count = 0

                results['threshold'] = threshold
                print_dict(logger, string=f'* BEST accuracy at epoch {current_epoch} with threshold {best_threshold}', dict=results)

                logger.info('Saving best model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                logger.info('Done saving...')
            else:
                print_dict(logger, string=f'* Current accuracy at epoch {current_epoch} with threshold {best_threshold}', dict=results)

                logger.info('Saving current model...')
                torch.save(model.state_dict(), os.path.join(log_dir, 'current.pth'))
                logger.info('Done saving...')

                if early_stop_count == args.train.early_stop:
                    logger.info('End.')
                    end_time = time.time()
                    logger.info(f'Done training at epoch {current_epoch}. Total time : {end_time-start_time}')     

                    print_dict(logger, string=f'** BEST RESULTS with threshold {best_threshold}', dict=best_results)

                    exit()
                early_stop_count += 1
                logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')

    
    print_dict(logger, string=f'** BEST RESULTS with threshold {best_threshold}', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

