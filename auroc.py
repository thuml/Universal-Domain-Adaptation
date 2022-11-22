
import argparse
import copy
import time
import datetime
import logging
import matplotlib.pyplot as plt

import easydict
import torch
import yaml
from torch import nn
from tqdm import tqdm

from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models import (
    uan, resnet
)
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything
from utils.evaluation import HScore
from utils.data import *

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

logger = logging.getLogger(__name__)

METHOD_TO_MODEL = {
    'fine_tuning' : resnet.ResNet,
    'uan' : uan.UAN,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--lr', type=float, default=None, help='Custom learning rate.')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold from training.')
    parser.add_argument('--min_threshold', type=float, default=0.0, help='Minimum threshold value.')
    parser.add_argument('--max_threshold', type=float, default=1.0, help='Maximum threshold value.')
    parser.add_argument('--method', type=str, default=None, help='Method to evaluate.')

    tmp_args = parser.parse_args()
    # lr = args.lr
    assert tmp_args.method is not None, f'Select a method from {METHOD_TO_MODEL.keys()}'

    config_file = tmp_args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    args.train.lr = tmp_args.lr
    args.threshold = tmp_args.threshold
    args.min_threshold = tmp_args.min_threshold
    args.max_threshold = tmp_args.max_threshold
    args.method = tmp_args.method

    return args, save_config

def test(model, dataloader, output_device, unknown_class):
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(dataloader, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = model.feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = model.classifier.forward(feature)
            domain_prob = model.discriminator_separate.forward(__)

            target_share_weight = model.get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0,
                                                        class_temperature=1.0)

            # OURS
            # shape (batch, )
            predictions = predict_prob.argmax(dim=-1)
            # pdb.set_trace()
            predictions[target_share_weight.reshape(-1) < args.test.w_0] = unknown_class
            metric.add_batch(predictions=predictions, references=label)
    
    results = metric.compute()
    return results


def test_with_threshold(model, dataloader, output_device, unknown_class, threshold):
    metric = HScore(unknown_class)

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

def cheating_test(model, dataloader, output_device, unknown_class, start=0.0, end=1.0, step=0.005):
    logger.info(f'Check threshold from {start} ~ {end} with step {step}')
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)


    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    max_logits_list = []

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

            max_logits_list.append(max_logits)

            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                unknown = (max_logits < threshold).squeeze()
                tmp_predictions[unknown] = unknown_class

                pdb.set_trace()
                metrics[index].add_batch(
                    predictions=tmp_predictions,
                    references=label
                )
    best_threshold = 0
    best_hscore = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = metrics[index].compute()
        # current_hscore = results['h_score']
        current_hscore = results['mean_accuracy']

        if current_hscore >= best_hscore:
            best_hscore = current_hscore
            best_threshold = threshold
            best_results = results
            print('\n*')

        print(threshold, '>', current_hscore)

    max_logits_list = torch.concat(max_logits_list)

    return best_results, best_threshold, max_logits_list

def get_all_predictions(model, dataloader, output_device, threshold=None):
    exit()
    labels = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(dataloader, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

        if threshold is not None:
            # MSP thresholding
            outputs = model.predict(**batch, threshold=threshold)
        else:
            outputs = model.predict(**batch)

        prediction = outputs['unk_logits']

        labels.append(label.cpu().detach().numpy())
        predictions.append(prediction.cpu().detach().numpy())

    # shape : (num_samples, )
    # concatenate all predictions and labels
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)

    return {
        'labels' : labels,
        'predictions' : predictions,
    }

def main(args, save_config):

    ## GPU SETTINGS ##
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]
    ## GPU SETTINGS ##

    
    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/{args.method}/{args.train.lr}'
    # init logger
    logger_init(logger, log_dir)
    ## LOGGINGS ##


    ## LOAD DATASETS ##
    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    MODEL_CLASS = METHOD_TO_MODEL.get(args.method)
    start_time = time.time()
    model = MODEL_CLASS(args=args, source_classes=source_classes).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    ## INIT MODEL ##
    ## LOAD MODEL ##
    state_dict_path = os.path.join(log_dir, 'best.pth')
    logger.info(f'LOAD MODEL from : {state_dict_path}')
    assert os.path.exists(state_dict_path)
    model.load_state_dict(torch.load(state_dict_path))
    ## LOAD MODEL ##
    # pdb.set_trace()


    
    logger.info('* Results from training ...')
    results = test_with_threshold(model, target_test_dl, output_device, unknown_class, args.threshold)
    print_dict(logger, string=f'Result from training with threshold {args.threshold}', dict=results)



    logger.info('* Cheating test for best h-score....')
    # results, best_threshold, max_logits_list = cheating_test(model, target_test_dl, output_device, unknown_class)
    results, best_threshold, max_logits_list = cheating_test(model, target_test_dl, output_device, unknown_class, start=0.7, end=0.73)
    print_dict(logger, string=f'BEST result with threshold {best_threshold}', dict=results)

    exit()
    total_count = len(max_logits_list)
    sorted_logits, indices = torch.sort(max_logits_list, descending=True)
    threshold_index = round(total_count * 0.95)
    threshold = sorted_logits[threshold_index]


    logger.info('* H-score @ 95 ...')
    results = test_with_threshold(model, target_test_dl, output_device, unknown_class, threshold)
    print_dict(logger, string=f'H-score @ 95 with threshold {threshold}', dict=results)


    ## PLOT RESULTS ##
    logger.info('PLOT RESULTS ...')
    plt.plot(list(range(0, total_count)), sorted_logits.cpu().numpy())

    plt.plot(threshold_index, threshold.cpu().item(), 'go', label=f'98% : {threshold.cpu().item()}')
    plt.axvline(x=threshold_index, color='g')

    # pdb.set_trace()
    
    best_index = ((sorted_logits > best_threshold).nonzero().squeeze()[-1].item())
    plt.plot(best_index, best_threshold, 'ro', label=f'Best : {best_threshold}')
    plt.axvline(x=best_index, color='r')


    train_index = ((sorted_logits > args.threshold).nonzero().squeeze()[-1].item())
    plt.plot(train_index, args.threshold, 'yo', label=f'Train : {args.threshold}')
    plt.axvline(x=train_index, color='y')

    plt.xlabel(f'Index')
    plt.ylabel('Threshold')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'figure.png'))
        
    pdb.set_trace()


    #####################
    #                   #
    #  CALCULATE AUROC  #
    #                   #
    #####################

    source_dl, target_known_dl, target_unknown_dl = get_auroc_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)






    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

