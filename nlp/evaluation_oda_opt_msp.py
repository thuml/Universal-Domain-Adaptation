
import time
import logging
import copy
import os

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pdb

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.covariance import LedoitWolf
import torch.nn.functional as F

from models import (
    bert,
    cmu, 
    dann, 
    ovanet, 
    uan,
    udalm
)
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders_for_oda

cudnn.benchmark = True
cudnn.deterministic = True

logger = logging.getLogger(__name__)

METHOD_TO_MODEL = {
    'fine_tuning' : bert.BERT,
    'dann' : dann.DANN,
    'uan' : uan.UAN,
    'cmu' : cmu.CMU,
    'ovanet' : ovanet.OVANET,
    'udalm' : udalm.UDALM,
}

THRESHOLDING_METHODS = ['fine_tuning', 'dann', 'uan', 'cmu', 'udalm']


def cheating_cosine_test(model, dataloader, output_dict, unknown_class, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    cosine_metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]
    # cosine_metrics = copy.deepcopy(maha_metrics)

    class_list = output_dict['all_classes']

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            # shape : (batch, hidden_dim)
            embeddings = model(**test_batch, embeddings_only=True)

            norm_embeddings = F.normalize(embeddings, dim=-1)

            ## COSINE ##
            # shape : (batch, hidden_dim) @ (hidden_dim, num_samples) -> (batch, num_samples)
            cosine_scores = norm_embeddings @ output_dict['norm_bank'].t()

            # shape : (batch, )
            cosine_score, max_indices = cosine_scores.max(-1)
            cosine_pred = output_dict['label_bank'][max_indices]

            # check for best threshold
            for index in range(num_thresholds):
                tmp_cosine_predictions = cosine_pred.clone().detach()

                threshold = thresholds[index]

                unknown = (cosine_score < threshold).squeeze()
                tmp_cosine_predictions[unknown] = unknown_class

                cosine_metrics[index].add_batch(
                    predictions=tmp_cosine_predictions,
                    references=labels
                )

    best_threshold = 0
    best_metric = 0
    best_results = None

    for index in range(num_thresholds):
        threshold = thresholds[index]

        results = cosine_metrics[index].compute()
        current_metric = results[metric_name] * 100

        if current_metric >= best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_results = results

    best_results['threshold'] = best_threshold
    

    return best_results

def prepare_ood(model, dataloader=None):
    bank = None
    label_bank = None
    for batch in tqdm(dataloader, desc='Generating training distribution'):
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        
        # shape : (batch, hidden_dim)
        pooled = model.forward(**batch, embeddings_only=True)

        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach()
        else:
            new_bank = pooled.clone().detach()
            new_label_bank = labels.clone().detach()
            bank = torch.cat([new_bank, bank], dim=0)
            label_bank = torch.cat([new_label_bank, label_bank], dim=0)


    # shape : (num_sample, hidden_dim)
    norm_bank = F.normalize(bank, dim=-1)
    # shape : (num_sample, hidden_dim)
    N, d = bank.size()
    # shape : (num_class, )
    all_classes = list(set(label_bank.tolist()))
    # shape : (num_class, hidden_dim)
    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()

    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    # shape : (num_class, hidden_dim)
    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    
    # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
    precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().cuda()

    return {
        # shape : (hidden_dim, hidden_dim)
        'class_var' : class_var,
        # shape : (num_class, hidden_dim)
        'class_mean' : class_mean,
        # shape : (num_samples, hidden_dim)
        'norm_bank' : norm_bank,
        # list of range(0, num_class)
        'all_classes' : all_classes,
        # shape : (num_class, )
        'label_bank' : label_bank,
    }


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

def cheating_test(model, dataloader, unknown_class, metric_name='total_accuracy', start=0.0, end=1.0, step=0.005):
    thresholds = list(np.arange(start, end, step))
    num_thresholds = len(thresholds)

    metric = HScore(unknown_class)

    print(f'Number of thresholds : {num_thresholds}')

    metrics = [copy.deepcopy(metric) for _ in range(num_thresholds)]

    max_logits_list = []

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, predictions = outputs['max_logits'], outputs['predictions']
            
            max_logits_list.append(max_logits)

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
    
    max_logits_list = torch.concat(max_logits_list)

    return best_results, max_logits_list


def eval(model, dataloader, unknown_class,):
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

def test(model, dataloader, unknown_class):
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


def main(args, save_config):
    seed_everything(args.train.seed)

    assert args.method_name in METHOD_TO_MODEL.keys()

    thresholding = True if args.method_name in THRESHOLDING_METHODS else False
    
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/{args.method_name}/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
    # init logger
    logger_init(logger, log_dir)
    ## LOGGINGS ##


    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')

    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATALOADER ##
    train_dataloader, _, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders_for_oda(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.test.batch_size, max_length=args.train.max_length)

    unknown_dataset = test_dataloader.dataset.filter(lambda sample: sample['labels'] == unknown_label)
    adaptable_dataset = test_dataloader.dataset.filter(lambda sample: sample['labels'] != unknown_label)

    
    data_collator = DataCollatorWithPadding(tokenizer)
    unknown_dataloader = DataLoader(unknown_dataset, collate_fn=data_collator, batch_size=args.test.batch_size, shuffle=False) 
    adaptable_dataloader = DataLoader(adaptable_dataset, collate_fn=data_collator, batch_size=args.test.batch_size, shuffle=False) 

    # pdb.set_trace()

    ## INIT MODEL ##
    logger.info(f'Init model {args.method_name} ...')
    METHOD = METHOD_TO_MODEL.get(args.method_name)
    start_time = time.time()
    model = METHOD(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
        max_train_step=100,     # dummy value
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

     
    model_dir = os.path.join(log_dir, 'best.pth')
    logger.info(f'Loading best model from : {model_dir}')
    model.load_state_dict(torch.load(model_dir))

    
    ####################
    #                  #
    #       Test       #
    #                  #
    ####################

    # # eval : source eval set
    # results = cheating_eval(model, eval_dataloader, unknown_label, is_cda=False, start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
            
    # print_dict(logger, string=f'\n\n** CHEATING SOURCE EVAL RESULT', dict=results)

    # eval_threshold = results['threshold']
    # # test : target test set
    # results = test_with_threshold(model, test_dataloader, unknown_label, eval_threshold)
    # print_dict(logger, string=f'\n\n** TARGET TEST RESULT with OPT-MSP threshold {eval_threshold}', dict=results)

    # if args.test.threshold is not None:
    #     # test : target test set
    #     results = test_with_threshold(model, test_dataloader, unknown_label, args.test.threshold)
    #     print_dict(logger, string=f'\n\n** TARGET TEST with fixed threshold {args.test.threshold}', dict=results)

    # cheating test : target test set
    results, max_logits_list = cheating_test(model, test_dataloader, unknown_label, metric_name='h_score', start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
    print_dict(logger, string=f'\n\n** OPTIMAL TARGET TEST RESULT', dict=results)


    # show results with threshold at 95%
    ratio = args.test.fpr_rate

    total_count = len(max_logits_list)
    sorted_logits, indices = torch.sort(max_logits_list, descending=True)
    threshold_index = round(total_count * ratio)
    threshold = sorted_logits[threshold_index]

    logger.info(f'* MSP @ {ratio} ...')
    results = test_with_threshold(model, test_dataloader, unknown_label, threshold)
    print_dict(logger, string=f'MSP @ {ratio} with threshold {threshold}', dict=results)


    output_dict = prepare_ood(model, dataloader=train_dataloader)

    best_results = cheating_cosine_test(model, test_dataloader, output_dict, unknown_label, metric_name='h_score', start=0.0, end=1.0, step=0.005)

    print_dict(logger, string=f'\n\n** CHEATING TARGET DOMAIN TEST RESULT USING COSINE SIMILARITY', dict=best_results)

    logger.info('Done.')


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

