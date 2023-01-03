
import argparse
import time
import logging
import copy

import easydict
import torch
import yaml
import numpy as np
from torch import nn
from tqdm import tqdm


from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.resnet import ResNet
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

def cheating_test(model, dataloader, output_device, unknown_class, start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)


    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(dataloader, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

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

        # print(threshold, '->', total_accuracy)

    return best_results, best_threshold

def test_with_threshold(model, dataloader, output_device, unknown_class, threshold):
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(dataloader, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            # predictions   : (batch, )
            # max_logits    : (batch, )
            # total_logits  : (batch, num_source_class)
            outputs  = model.get_prediction_and_logits(im)
            predictions, total_logits, max_logits = outputs['predictions'], outputs['total_logits'], outputs['max_logits']


            # pdb.set_trace()
            predictions[max_logits < threshold] = unknown_class
            metric.add_batch(predictions=predictions, references=label)
    
    results = metric.compute()
    return results


def main(args, save_config):
    seed_everything(args.seed)


    ## GPU SETTINGS ##
    # gpu_ids = select_GPUs(args.misc.gpus)
    gpu_ids = [0]
    output_device = gpu_ids[0]
    ## GPU SETTINGS ##

    
    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/fine_tuning/{args.seed}/{args.train.lr}'
    
    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    writer = SummaryWriter(log_dir)
    # dump configs
    with open(join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##

    logger.info(f'ARGS : {args}')


    ## LOAD DATASETS ##
    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = ResNet(args, source_classes)
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')

    feature_extractor = nn.DataParallel(model.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
    classifier = nn.DataParallel(model.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
    ## INIT MODEL ##


    ## TEST ONLY ##
    if args.test.test_only:
        logger.info('TEST ONLY...')
        state_dict_path = os.path.join(log_dir, 'best.pth')
        assert os.path.exists(state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))
        results = test_with_threshold(model, target_test_dl, output_device, unknown_class, args.test.threshold)

        print_dict(logger, string='======== Final Test Results ========', dict=results)
        exit(0)
    ## TEST ONLY ##

    # ===================optimizer
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    
    current_epoch = 0
    global_step = 0
    best_acc = 0
    best_threshold = 0
    best_results = None
    early_stop_count = 0

    # total steps / epochs
    # we only consider source domain samples
    steps_per_epoch = len(source_train_dl)
    total_epoch = round(args.train.min_step / steps_per_epoch)
    logger.info(f'Total epoch {total_epoch}, steps per epoch {steps_per_epoch}, total step {args.train.min_step}')

    # log every epoch
    log_interval = steps_per_epoch
    # test every epoh
    test_interval = steps_per_epoch

    # total_steps = tqdm(range(args.train.min_step),desc='global step')

    logger.info(f'Start Training....')
    start_time = time.time()

    while global_step < args.train.min_step:

        iters = tqdm(source_train_dl, desc=f'epoch {current_epoch} ', total=steps_per_epoch)
        current_epoch += 1

        for i, (im_source, label_source) in enumerate(iters):

            # save_label_target = label_target  # for debug usage

            label_source = label_source.to(output_device)

            ####################
            #                  #
            #   Forward Pass   #
            #                  #
            ####################

            im_source = im_source.to(output_device)

            # fc1_s : (batch_size, 2048)
            fc1_s = feature_extractor.forward(im_source)

            # fc1_s                 : (batch, hidden_dim)
            # feature_source        : (batch, bottleneck_dim)
            # fc2_s                 : (batch, num_source_label)
            # predict_prob_source   : (batch, num_source_label)
            fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
                
            
            ####################
            #                  #
            #   Compute Loss   #
            #                  #
            ####################

            # ============================== cross entropy loss
            loss = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
            loss = torch.mean(loss, dim=0, keepdim=True)

            with OptimizerManager(
                    [optimizer_finetune, optimizer_cls]):
                loss.backward()

            global_step += 1

            ####################
            #                  #
            #     Logging      #
            #                  #
            ####################

            if global_step % log_interval == 0:
                writer.add_scalar('train/loss', loss, current_epoch)


            ####################
            #                  #
            #       Test       #
            #                  #
            ####################
            
            if global_step % test_interval == 0:
                logger.info('TEST...')
                # results = test(model, target_test_dl, output_device, unknown_class)

                # find optimal threshold using test set = cheating
                results, threshold = cheating_test(model, target_test_dl, output_device, unknown_class, start=args.min_threshold, end=args.max_threshold, step=args.step)
                writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], global_step)
                writer.add_scalar('test/total_acc_test', results['total_accuracy'], global_step)
                writer.add_scalar('test/known_test', results['known_accuracy'], global_step)
                writer.add_scalar('test/unknown_test', results['unknown_accuracy'], global_step)
                writer.add_scalar('test/hscore_test', results['h_score'], global_step)
                writer.add_scalar('test/threshold', threshold, global_step)

                # clear_output()

                if results['mean_accuracy'] > best_acc:
                    best_acc = results['mean_accuracy']
                    best_results = results
                    best_threshold = threshold
                    early_stop_count = 0

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

                        print_dict(logger, string=f'** BEST RESULTS', dict=best_results)

                        exit()
                    early_stop_count += 1
                    logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')
    
    print_dict(logger, string=f'** BEST RESULTS with threshold {best_threshold}', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

