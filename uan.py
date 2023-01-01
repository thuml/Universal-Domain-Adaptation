import os
import argparse
import time
import datetime
import logging

import easydict
import torch
import yaml
from torch import nn
from tqdm import tqdm

from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.uan import UAN
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

def test(model, dataloader, output_device, unknown_class):
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
            predictions, _, max_logits = outputs['predictions'], outputs['total_logits'], outputs['max_logits']

            # pdb.set_trace()
            predictions[max_logits < args.test.w_0] = unknown_class
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
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/uan/{args.seed}/{args.train.lr}'
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
    model = UAN(args, source_classes)
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')

    feature_extractor = nn.DataParallel(model.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
    classifier = nn.DataParallel(model.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
    discriminator = nn.DataParallel(model.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
    discriminator_separate = nn.DataParallel(model.discriminator_separate, device_ids=gpu_ids, output_device=output_device).train(True)
    ## INIT MODEL ##


    ## TEST ONLY ##
    if args.test.test_only:
        logger.info('TEST ONLY...')
        state_dict_path = os.path.join(log_dir, 'best.pth')
        assert os.path.exists(state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))
        results = test(model, target_test_dl, output_device, unknown_class)

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
    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_discriminator_separate = OptimWithSheduler(
        optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)

    current_epoch = 0
    global_step = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    # total steps / epochs
    steps_per_epoch = max(len(source_train_dl), len(target_train_dl))
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

        iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {current_epoch} ', total=steps_per_epoch)
        current_epoch += 1

        for i, ((im_source, label_source), (im_target, _)) in enumerate(iters):

            # save_label_target = label_target  # for debug usage

            label_source = label_source.to(output_device)
            # label_target = label_target.to(output_device)
            # label_target = torch.zeros_like(label_target)

            ####################
            #                  #
            #   Forward Pass   #
            #                  #
            ####################

            im_source = im_source.to(output_device)
            im_target = im_target.to(output_device)

            # fc1_s : (batch_size, 2048)
            fc1_s = feature_extractor.forward(im_source)
            fc1_t = feature_extractor.forward(im_target)

            # fc1_s                 : (batch, hidden_dim)
            # feature_source        : (batch, bottleneck_dim)
            # fc2_s                 : (batch, num_source_label)
            # predict_prob_source   : (batch, num_source_label)
            fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
            fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

            # shape : (batch, 1)
            domain_prob_discriminator_source = discriminator.forward(feature_source)
            # shape : (batch, 1)
            domain_prob_discriminator_target = discriminator.forward(feature_target)

            # shape : (batch, 1)
            domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
            # shape : (batch, 1)
            domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

            # shape : (batch, 1)
            source_share_weight = model.get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
            source_share_weight = model.normalize_weight(source_share_weight)
            # shape : (batch, 1)
            target_share_weight = model.get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
            target_share_weight = model.normalize_weight(target_share_weight)
                
            
            ####################
            #                  #
            #   Compute Loss   #
            #                  #
            ####################

            adv_loss = torch.zeros(1, 1).to(output_device)
            adv_loss_separate = torch.zeros(1, 1).to(output_device)

            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)
            tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
            adv_loss += torch.mean(tmp, dim=0, keepdim=True)

            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))

            # ============================== cross entropy loss
            ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
            ce = torch.mean(ce, dim=0, keepdim=True)

            with OptimizerManager(
                    [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_discriminator_separate]):
                loss = ce + adv_loss + adv_loss_separate
                loss.backward()

            global_step += 1

            ####################
            #                  #
            #     Logging      #
            #                  #
            ####################

            if global_step % log_interval == 0:
                writer.add_scalar('train/adv_loss', adv_loss, current_epoch)
                writer.add_scalar('train/ce', ce, current_epoch)
                writer.add_scalar('train/adv_loss_separate', adv_loss_separate, current_epoch)


            ####################
            #                  #
            #       Test       #
            #                  #
            ####################
            
            if global_step % test_interval == 0:
                logger.info('TEST...')
                results = test(model, target_test_dl, output_device, unknown_class)
                writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], global_step)
                writer.add_scalar('test/total_acc_test', results['total_accuracy'], global_step)
                writer.add_scalar('test/known_test', results['known_accuracy'], global_step)
                writer.add_scalar('test/unknown_test', results['unknown_accuracy'], global_step)
                writer.add_scalar('test/hscore_test', results['h_score'], global_step)

                # clear_output()

                if results['mean_accuracy'] > best_acc:
                    best_acc = results['mean_accuracy']
                    best_results = results
                    early_stop_count = 0

                    print_dict(logger, string=f'* BEST accuracy at epoch {current_epoch}', dict=results)

                    logger.info('Saving best model...')
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
                    logger.info('Done saving...')
                else:
                    print_dict(logger, string=f'* Current accuracy at epoch {current_epoch}', dict=results)

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

