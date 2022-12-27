
import argparse
import time
import logging
import copy
import os

import easydict
import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
import pdb

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

from models.cmu import CMU
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders_for_oda, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)


def cheating_test(model, dataloader, unknown_class, is_cda=False, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            weight, predictions = outputs['max_logits'], outputs['predictions']      

            # check for best threshold
            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                # predict "unknown" for opda setting
                if not is_cda:
                    unknown = (weight < threshold).squeeze()
                    tmp_predictions[unknown] = unknown_class

                metrics[index].add_batch(
                    predictions=tmp_predictions,
                    references=labels
                )

    best_threshold = 0
    best_metric = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = metrics[index].compute()
        current_metric = results[metric_name] * 100

        if current_metric >= best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_results = results

    best_results['threshold'] = best_threshold

    return best_results


def test_for_cda(model, dataloader):
    metric = Accuracy()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='Testing CDA'):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()

    return results

def test_with_threshold(model, dataloader, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            weight, predictions = outputs['max_logits'], outputs['predictions']      

            predictions[weight <= threshold] = unknown_class
            
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

    return results

def eval_with_threshold(model, dataloader, is_cda, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    metric = Accuracy()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            weight, predictions = outputs['max_logits'], outputs['predictions']

            if is_cda:
                predictions[weight <= threshold] = unknown_class
            
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

    return results

def main(args, save_config):
    seed_everything(args.train.seed)


    source_domain = None
    target_domain = None
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/cmu/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'

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
    train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders_for_oda(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.train.batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain)

    num_step_per_epoch = max(len(train_dataloader), len(train_unlabeled_dataloader))
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = CMU(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
        max_train_step=total_step
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##


    ## OPTIMIZER & SCHEDULER ##  
    base_optimizer = optim.AdamW(model.model.parameters(), lr=args.train.lr / 10)
    bottleneck_optimizer = optim.AdamW(model.classifier.bottleneck.parameters(), lr=args.train.lr)
    classifier_params = [{"params": model.classifier.fc.parameters()}, {"params": model.classifier.fc2.parameters()},
            {"params": model.classifier.fc3.parameters()}, {"params": model.classifier.fc4.parameters()},
            {"params": model.classifier.fc5.parameters()}]
    classifier_optimizer = optim.AdamW(classifier_params, lr=args.train.lr * 5)
    disc_optimizer = optim.AdamW(model.discriminator.parameters(), lr=args.train.lr)

    num_warmup_steps = int(total_step * 0.3)

    base_lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=base_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    bottleneck_lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=bottleneck_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    classifier_lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=classifier_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    disc_lr_scheduler = get_scheduler(
        name=args.train.lr_scheduler_type,
        optimizer=disc_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_step
    )
    ## OPTIMIZER & SCHEDULER ##  
    
    global_step = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    # data iter
    source_iter = ForeverDataIterator(train_dataloader)
    target_iter = ForeverDataIterator(train_unlabeled_dataloader)


    # CE-loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    # BCE-loss for domain classification
    bce = nn.BCELoss().cuda()

    logger.info('Start main training...')

    current_epoch = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
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
                base_optimizer.zero_grad()
                bottleneck_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                disc_optimizer.zero_grad()

                ####################
                #                  #
                #     Load Data    #
                #                  #
                ####################

                ## source to cuda
                source_batch = next(source_iter)
                source_batch = {k: v.cuda() for k, v in source_batch.items()}
                source_labels = source_batch['labels']

                ## target to cuda
                target_batch = next(target_iter)
                target_batch = {k: v.cuda() for k, v in target_batch.items()}

                ####################
                #                  #
                #   Forward Pass   #
                #                  #
                ####################

                ## source 
                source_output = model(**source_batch)

                source_logits, source_1, source_2, source_3, source_4, source_5, source_domain_output = \
                    source_output['logits'], source_output['fc2_1'], source_output['fc2_2'], source_output['fc2_3'], \
                        source_output['fc2_4'], source_output['fc2_5'], source_output['domain_output']

                ## target 
                target_output = model(**target_batch)

                target_domain_output = target_output['domain_output']

                
                ####################
                #                  #
                #   Compute Loss   #
                #                  #
                ####################

                source_adv_loss = bce(source_domain_output, torch.ones_like(source_domain_output))
                target_adv_loss = bce(target_domain_output, torch.zeros_like(target_domain_output))
                adv_loss = source_adv_loss + target_adv_loss


                ce_loss1 = ce(source_1, source_labels)
                ce_loss2 = ce(source_2, source_labels)
                ce_loss3 = ce(source_3, source_labels)
                ce_loss4 = ce(source_4, source_labels)
                ce_loss5 = ce(source_5, source_labels)
                ce_loss = (ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5) / 5

                # total loss
                loss = adv_loss + ce_loss

                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/ce_loss', ce_loss, global_step)
                writer.add_scalar('train/adv_loss', adv_loss, global_step)

                # backward, optimization
                loss.backward()

                base_optimizer.step()
                bottleneck_optimizer.step()
                classifier_optimizer.step()
                disc_optimizer.step()

                base_lr_scheduler.step()
                bottleneck_lr_scheduler.step()
                classifier_lr_scheduler.step()
                disc_lr_scheduler.step()
            
            ####################
            #                  #
            #     Evaluate     #
            #                  #
            ####################
            
            logger.info(f'Evaluate model at epoch {current_epoch} ...')

            # find optimal threshold from evaluation set (source domain) -> sub-optimal threshold
            results = eval_with_threshold(model, eval_dataloader, False, unknown_label, args.test.threshold)
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
    

    logger.info('TEST WITH "UNKNOWN" CLASS.')
    best_threshold = best_results['threshold'] if best_results is not None else args.test.threshold
    results = test_with_threshold(model, test_dataloader, unknown_label, best_threshold)
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 0)

    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

    # Find optimal threshold from test set (Cheating)
    # find model with the best h-score
    results = cheating_test(model, test_dataloader, unknown_label, metric_name='h_score', start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
    # write to tensorboard
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 1)
    print_dict(logger, string=f'\n\n** CHEATING TARGET DOMAIN TEST RESULT', dict=results)


    logger.info('Done.')
                



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

