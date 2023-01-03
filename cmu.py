
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

from models.cmu import CMU
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

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L226
def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L179
def get_consistency(y_1, y_2, y_3, y_4, y_5):
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
def get_entropy(y_1, y_2, y_3, y_4, y_5):
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
def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    return entropy

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L210
def get_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, indice_1 = torch.max(y_1, 1)
    conf_2, indice_2 = torch.max(y_2, 1)
    conf_3, indice_3 = torch.max(y_3, 1)
    conf_4, indice_4 = torch.max(y_4, 1)
    conf_5, indice_5 = torch.max(y_5, 1)
    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence

def test(dataloader, model, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    metric = HScore(unknown_class)

    model.eval()
   
    with torch.no_grad():
        for i, (im, labels) in enumerate(dataloader):
            im = im.cuda()
            labels = labels.cuda()

            feature = model.feature_extractor(im)
            feature, __, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob = model.classifier(feature)

            entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
            consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
            # confidence, indices = torch.max(predict_prob, dim=1)
            confidence = get_confidence(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()

            entropy = norm(torch.tensor(entropy))
            consistency = norm(torch.tensor(consistency))
            confidence = norm(torch.tensor(confidence))

            # TODO : re?
            weight = (1 - entropy + 1 - consistency + confidence) / 3
            # weight = (entropy + consistency) / 2
            # threshold = torch.mean(weight).cuda()

            # shape : (batch, )
            predictions = predict_prob.argmax(dim=-1)

            predictions[weight <= threshold] = unknown_class
            
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    return results

    
def main(args, save_config):
    seed_everything(args.seed)
    
    ## GPU SETTINGS ##
    # gpu_ids = select_GPUs(args.misc.gpus)
    # TODO : remove?
    gpu_ids = [0]
    output_device = gpu_ids[0]
    ## GPU SETTINGS ##

    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/cmu/{args.seed}/{args.train.lr}'
    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    if not args.test.test_only:
        writer = SummaryWriter(log_dir)
    # dump configs
    with open(join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##

    logger.info(f'ARGS : {args}')


    ## LOAD DATASETS ##
    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

    # dataloaders for ensemble
    dl1, dl2, dl3, dl4, dl5 = esem_dataloader(args, source_classes)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = CMU(args, source_classes).cuda()
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
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(model.feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay,
                momentum=args.train.momentum, nesterov=True), scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(model.classifier.bottleneck.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
                momentum=args.train.momentum, nesterov=True), scheduler)
    fc_para = [{"params": model.classifier.fc.parameters()}, {"params": model.classifier.fc2.parameters()},
            {"params": model.classifier.fc3.parameters()}, {"params": model.classifier.fc4.parameters()},
            {"params": model.classifier.fc5.parameters()}]
    optimizer_fc = OptimWithSheduler(
        optim.SGD(fc_para, lr=args.train.lr * 5, weight_decay=args.train.weight_decay,
                momentum=args.train.momentum, nesterov=True), scheduler)
    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(model.discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
                momentum=args.train.momentum, nesterov=True), scheduler)    
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

    # forever iterator for ensemble
    iter1 = ForeverDataIterator(dl1)
    iter2 = ForeverDataIterator(dl2)
    iter3 = ForeverDataIterator(dl3)
    iter4 = ForeverDataIterator(dl4)
    iter5 = ForeverDataIterator(dl5)

    # CE-loss
    ce = nn.CrossEntropyLoss().cuda()
    bce = nn.BCELoss().cuda()

    logger.info('Start main training...')

    current_epoch = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
    for current_step in tqdm(range(args.train.min_step), desc='Train Model'):
        
        ####################
        #                  #
        #       Train      #
        #                  #
        ####################
        
        model.train()

        # load data
        im_s, label_s = next(source_iter)
        im1, label1 = next(iter1)
        im2, label2 = next(iter2)
        im3, label3 = next(iter3)
        im4, label4 = next(iter4)
        # im5, label5 = next(iter5)
        im_t, label_t = next(target_iter)

        # to cuda
        im_s, label_s = im_s.cuda(), label_s.cuda()
        im1, label1 = im1.cuda(), label1.cuda()
        im2, label2 = im2.cuda(), label2.cuda()
        im3, label3 = im3.cuda(), label3.cuda()
        im4, label4 = im4.cuda(), label4.cuda()
        # im5, label5 = im5.cuda(), label5.cuda()
        im_t, label_t = im_t.cuda(), label_t.cuda()

        # extract feature
        fc1_s = model.feature_extractor.forward(im_s)
        fc1_s2 = model.feature_extractor.forward(im1)
        fc1_s3 = model.feature_extractor.forward(im2)
        fc1_s4 = model.feature_extractor.forward(im3)
        fc1_s5 = model.feature_extractor.forward(im4)
        fc1_t = model.feature_extractor.forward(im_t)

        fc1_s, feature_source, fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5, predict_prob_source = model.classifier.forward(fc1_s)
        fc1_s2, feature_source2, fc2_s_2, fc2_s2_2, fc2_s3_2, fc2_s4_2, fc2_s5_2, predict_prob_source2 = \
            model.classifier.forward(fc1_s2)
        fc1_s3, feature_source3, fc2_s_3, fc2_s2_3, fc2_s3_3, fc2_s4_3, fc2_s5_3, predict_prob_source3 = \
            model.classifier.forward(fc1_s3)
        fc1_s4, feature_source4, fc2_s_4, fc2_s2_4, fc2_s3_4, fc2_s4_4, fc2_s5_4, predict_prob_source4 = \
            model.classifier.forward(fc1_s4)
        fc1_s5, feature_source5, fc2_s_5, fc2_s2_5, fc2_s3_5, fc2_s4_5, fc2_s5_5, predict_prob_source5 = \
            model.classifier.forward(fc1_s5)
        fc1_t, feature_target, fc2_t, fc2_t2, fc2_t3, fc2_t4, fc2_t5, predict_prob_target = model.classifier.forward(fc1_t)


        domain_prob_discriminator_source = model.discriminator.forward(feature_source)
        domain_prob_discriminator_target = model.discriminator.forward(feature_target)

        # entropy = get_entropy(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
        # consistency = get_consistency(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5).detach()
        # confidence, indices = torch.max(predict_prob_target, dim=1)

        # adv loss
        source_adv_loss = bce(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        target_adv_loss = bce(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))

        # classificaiton loss
        # ce_loss1 = ce(fc2_s, label1)
        # ce_loss2 = ce(fc2_s2_2, label2)
        # ce_loss3 = ce(fc2_s3_3, label3)
        # ce_loss4 = ce(fc2_s4_4, label4)
        # ce_loss5 = ce(fc2_s5_5, label5)
        
        ce_loss1 = ce(fc2_s, label_s)
        ce_loss2 = ce(fc2_s2_2, label1)
        ce_loss3 = ce(fc2_s3_3, label2)
        ce_loss4 = ce(fc2_s4_4, label3)
        ce_loss5 = ce(fc2_s5_5, label4)
        ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5

        loss = ce_loss + source_adv_loss + target_adv_loss

        # backward()
        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_fc, optimizer_discriminator]):
            loss.backward()

        ####################
        #                  #
        #     Logging      #
        #                  #
        ####################
        writer.add_scalar('train/loss', loss, current_step)
        writer.add_scalar('train/source_adv_loss', source_adv_loss, current_step)
        writer.add_scalar('train/target_adv_loss', target_adv_loss, current_step)
        writer.add_scalar('train/ce_loss', ce_loss, current_step)

        ####################
        #                  #
        #       Test       #
        #                  #
        ####################

        if current_step % test_interval == 0:
            logger.info(f'TEST at epoch {current_epoch} ...')
            results = test(target_test_dl, model, unknown_class, args.test.threshold)
            writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], current_epoch)
            writer.add_scalar('test/total_acc_test', results['total_accuracy'], current_epoch)
            writer.add_scalar('test/known_test', results['known_accuracy'], current_epoch)
            writer.add_scalar('test/unknown_test', results['unknown_accuracy'], current_epoch)
            writer.add_scalar('test/hscore_test', results['h_score'], current_epoch)

            if results['mean_accuracy'] > best_acc:
                best_acc = results['mean_accuracy']
                best_results = results
                early_stop_count = 0

                print_dict(logger, string=f'* Best Result at epoch {current_epoch}', dict=results)

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

            current_epoch += 1

    print_dict(logger, string=f'** BEST RESULTS', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

