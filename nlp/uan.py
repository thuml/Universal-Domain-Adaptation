
import time
import logging
import copy
import os

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

from models.uan import UAN
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True

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

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            # check for best threshold
            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                # predict "unknown" for opda setting
                if not is_cda:
                    unknown = (max_logits < threshold).squeeze()
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

def cheating_eval(model, dataloader, unknown_class, is_cda=False, start=0.0, end=1.0, step=0.005):
    logger.info(f'Evaluation with best threshold : {start} ~ {end}')
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = Accuracy()

    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            # check for best threshold
            for index in range(num_thresholds):
                tmp_predictions = predictions.clone().detach()
                threshold = thresholds[index]

                # predict "unknown" for opda setting
                if not is_cda:
                    unknown = (max_logits < threshold).squeeze()
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
        current_metric = results['accuracy'] * 100

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
    
def eval_with_threshold(model, dataloader, unknown_class, threshold):
    logger.info(f'Test with threshold {threshold}')
    metric = Accuracy()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            # pdb.set_trace()
            predictions[max_logits < threshold] = unknown_class
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

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

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            # pdb.set_trace()
            predictions[max_logits < threshold] = unknown_class
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

    return results


def main(args, save_config):
    seed_everything(args.train.seed)
        
    if args.dataset.num_source_class == args.dataset.num_common_class:
        is_cda = True
        split = 'cda'
    else:
        is_cda = False
        split = 'opda'

    # amazon reviews data -> always cda
    if 'source_domain' in args.dataset:
        source_domain = args.dataset.source_domain
        target_domain = args.dataset.target_domain
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/uan/{source_domain}-{target_domain}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    # clinc, massive, trec
    else:
        source_domain = None
        target_domain = None
        coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
        ## LOGGINGS ##
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/uan/{split}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    

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
    train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.train.batch_size, max_length=args.train.max_length, source=source_domain, target=target_domain)

    num_step_per_epoch = max(len(train_dataloader), len(train_unlabeled_dataloader))
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UAN(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
        max_train_step=total_step,
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

    global_step = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0
    best_threshold = 0

    
    # data iter
    source_iter = ForeverDataIterator(train_dataloader)
    target_iter = ForeverDataIterator(train_unlabeled_dataloader)

    # cross-entropy loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    # bce loss for domain classification
    bce = nn.BCELoss().cuda()

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

                ## source to cuda
                target_batch = next(target_iter)
                target_batch = {k: v.cuda() for k, v in target_batch.items()}
                
                ####################
                #                  #
                #   Forward Pass   #
                #                  #
                ####################

                ## source ##
                source_outputs = model(**source_batch)
                # d : discriminator
                # d_0 : discriminator_separate
                source_logits, source_before_softmax, source_d, source_d_0 = source_outputs['logits'], source_outputs['before_softmax'], source_outputs['d'], source_outputs['d_0']

                # shape : (batch, 1)
                source_share_weight = model.get_source_share_weight(source_d_0, source_before_softmax, domain_temperature=1.0, class_temperature=10.0)
                source_share_weight = model.normalize_weight(source_share_weight)

                ## target ##
                target_outputs = model(**target_batch)
                # d : discriminator
                # d_0 : discriminator_separate
                target_logits, target_before_softmax, target_d, target_d_0 = target_outputs['logits'], target_outputs['before_softmax'], target_outputs['d'], target_outputs['d_0']

                # shape : (batch, 1)
                target_share_weight = model.get_target_share_weight(target_d_0, target_before_softmax, domain_temperature=1.0, class_temperature=1.0)
                target_share_weight = model.normalize_weight(target_share_weight)

                ####################
                #                  #
                #   Compute Loss   #
                #                  #
                ####################

                source_adv_loss = torch.mean(source_share_weight * bce(source_d, torch.ones_like(source_d)), dim=0)
                target_adv_loss = torch.mean(target_share_weight * bce(target_d, torch.zeros_like(target_d)), dim=0)
                adv_loss = source_adv_loss + target_adv_loss
                
                source_adv_loss_separate = bce(source_d_0, torch.ones_like(source_d_0))
                target_adv_loss_separate = bce(target_d_0, torch.zeros_like(target_d_0))
                adv_loss_separate = source_adv_loss_separate + target_adv_loss_separate

                ce_loss = ce(source_logits, source_labels)
                # total loss
                loss = ce_loss + adv_loss + adv_loss_separate

                # pdb.set_trace()

                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/ce_loss', ce_loss, global_step)
                writer.add_scalar('train/adv_loss', adv_loss, global_step)
                writer.add_scalar('train/adv_loss_separate', adv_loss_separate, global_step)

                # backward, optimization
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            ####################
            #                  #
            #     Evaluate     #
            #                  #
            ####################
            
            logger.info(f'Evaluate model at epoch {current_epoch} ...')

            # find optimal threshold from evaluation set (source domain) -> sub-optimal threshold
            # results = eval_with_threshold(model, test_dataloader, unknown_label, args.test.threshold)
            results = cheating_eval(model, eval_dataloader, unknown_label, is_cda, start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
    
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

    best_threshold = best_results['threshold']
            
    logger.info('Test model...')
    if is_cda:
        logger.info('TEST ON CDA SETTING.')
        results = test_for_cda(model, test_dataloader)
        for k,v in results.items():
            writer.add_scalar(f'test/{k}', v, 0)

        print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)
    else:
        logger.info('TEST WITH "UNKNOWN" CLASS.')
        results = test_with_threshold(model, test_dataloader, unknown_label, best_threshold)
        # for optimal uan method, we use a fixed threshold = -0.5
        # results = test_with_threshold(model, test_dataloader, unknown_label, args.test.threshold)
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

