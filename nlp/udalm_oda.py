
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import (
    AutoTokenizer,
    get_scheduler,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)

from models.udalm import UDALM
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore,Accuracy
from utils.data import get_datasets_for_oda, ForeverDataIterator
cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

def cheating_eval(model, dataloader, unknown_class, is_cda=False, metric_name='accuracy', start=0.0, end=1.0, step=0.005):
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

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']

            # pdb.set_trace()
            predictions[max_logits < threshold] = unknown_class
            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    results['threshold'] = threshold

    return results


def cheating_test(model, dataloader, unknown_class, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
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

def main(args, save_config):
    seed_everything(args.train.seed)

    source_domain = None
    target_domain = None
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
    mlm_dir = f'{args.log.output_dir}/{args.dataset.name}/udalm/mlm/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/0.0001'
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/udalm/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'

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
    train_data, train_unlabeled_data, eval_data, test_data, source_test_data = get_datasets_for_oda(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, source=source_domain, target=target_domain)

    n = len(train_data)
    m = len(train_unlabeled_data)
    cls_weight = n / (n + m)

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=512, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result

    # TOKENIZE
    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    eval_dataset = eval_data.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Running tokenizer on source eval dataset",
    )
    test_dataset = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )
    source_test_dataset = source_test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=source_test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )
    
    # DataLoaders creation :
    mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.train.mlm_probability)
    data_collator = DataCollatorWithPadding(tokenizer)

    # for training
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=mlm_data_collator, batch_size=args.train.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)
    # for evaluation
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=False)   
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=False) 

    num_step_per_epoch = max(len(train_unlabeled_dataloader), len(train_dataloader))
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')

    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UDALM(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

    ## LOAD 
    logger.info(f'Loading MLM pre-trained model from : {mlm_dir}')
    model.load_state_dict(torch.load(os.path.join(mlm_dir, 'best.pth')))


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
    best_acc = -1

    # data iter
    source_iter = ForeverDataIterator(train_dataloader)
    target_iter = ForeverDataIterator(train_unlabeled_dataloader)

    # CE-loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    
    ## START TRAINING ##
    logger.info(f'Start Training....')

    mlm_weight = 1 - cls_weight
    logger.info(f'TRAIN WITH CLS WEIGHT : {cls_weight},  MLM WEIGHT : {mlm_weight}')

    start_time = time.time()
    for current_epoch in range(1, args.train.num_train_epochs+1):
        model.train()

        epoch_loss = 0
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

            ## target to cuda
            target_batch = next(target_iter)
            target_batch = {k: v.cuda() for k, v in target_batch.items()}
            target_labels = target_batch['labels']

        
            ####################
            #                  #
            #   Forward Pass   #
            #                  #
            ####################

            ## source forward
            source_outputs = model(**source_batch, is_source=True)
            # shape : (batch, length, vocab_size)
            source_logits = source_outputs['logits']

            del source_outputs
            del source_batch

            ## target forward
            target_outputs = model(**target_batch, is_source=False)
            # shape : (batch, length, vocab_size)
            target_logits = target_outputs['logits']

            del target_outputs
            del target_batch

            ####################
            #                  #
            #   Compute Loss   #
            #                  #
            ####################

            cls_loss = ce(source_logits, source_labels)
            mlm_loss = ce(target_logits.view(-1, tokenizer.vocab_size), target_labels.view(-1))
            
            loss = cls_weight * cls_loss + mlm_weight * mlm_loss
            
            # write to tensorboard
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/cls_loss', cls_loss, global_step)
            writer.add_scalar('train/mlm_loss', mlm_loss, global_step)
            
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
        results = cheating_eval(model, eval_dataloader, unknown_label, start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
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

