
import time
import logging
import copy
import os

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pdb

from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import roc_curve, auc, roc_auc_score

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

# get all maximum logits values for calculating auroc
def get_all_predictions(model, dataloader, unknown_class):
    one_hot_labels_list = []
    labels_list = []
    predictions_list = []
    logits_list = []
    
    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']
            
            outputs = model(**test_batch)

            predictions, total_logits, max_logits = outputs['predictions'], outputs['logits'], outputs['max_logits']

            # for in-domain <-> adaptable-domain (no unknown class)
            # shape : (batch, num_source_class)
            one_hot_label = one_hot(labels, num_classes=unknown_class) if unknown_class not in labels else None


            labels_list.append(labels.cpu().detach().numpy())
            if one_hot_label is not None:
                one_hot_labels_list.append(one_hot_label.cpu().detach().numpy())
            predictions_list.append(max_logits.cpu().detach().numpy())
            logits_list.append(total_logits.cpu().detach().numpy())

    # shape : (num_samples, )
    # concatenate all predictions and labels
    labels = np.concatenate(labels_list)
    one_hot_labels = np.concatenate(one_hot_labels_list) if len(one_hot_labels_list) > 0 else None
    predictions = np.concatenate(predictions_list)
    logits = np.concatenate(logits_list)

    return {
        'labels' : labels,
        'one_hot_labels' : one_hot_labels,
        'predictions' : predictions,
        'logits' : logits,
    }


# calculate auroc
def calculate_auroc(labels, predictions, unknown_index):
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=unknown_index)
    roc_auc = auc(fpr, tpr) * 100

    return roc_auc

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
    _, _, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders_for_oda(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.test.batch_size, max_length=args.train.max_length)

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
    
    if thresholding:
        # # eval : source eval set
        # results = eval_with_threshold(model, eval_dataloader, unknown_label, args.test.threshold)
        # print_dict(logger, string=f'\n\n** SOURCE EVAL RESULT', dict=results)

        # # eval : source test set
        # results = eval_with_threshold(model, source_test_dataloader, unknown_label, args.test.threshold)
        # print_dict(logger, string=f'\n\n** SOURCE TEST RESULT', dict=results)

        # test : target test set
        results = test_with_threshold(model, test_dataloader, unknown_label, args.test.threshold)
        print_dict(logger, string=f'\n\n** TARGET TEST RESULT', dict=results)

        # cheating test : target test set
        results, max_logits_list = cheating_test(model, test_dataloader, unknown_label, metric_name='h_score', start=args.test.min_threshold, end=args.test.max_threshold, step=args.test.step)
        print_dict(logger, string=f'\n\n** OPTIMAL TARGET TEST RESULT', dict=results)

        # show results with threshold at 95%
        total_count = len(max_logits_list)
        sorted_logits, indices = torch.sort(max_logits_list, descending=True)
        threshold_index = round(total_count * 0.95)
        threshold = sorted_logits[threshold_index]

        logger.info('* H-score @ 95 ...')
        results = test_with_threshold(model, test_dataloader, unknown_label, threshold)
        print_dict(logger, string=f'H-score @ 95 with threshold {threshold}', dict=results)
    
    else:
        # eval : source eval set
        results = eval(model, eval_dataloader, unknown_label)
        print_dict(logger, string=f'\n\n** SOURCE EVAL RESULT', dict=results)

        # eval : source test set
        results = eval(model, source_test_dataloader, unknown_label)
        print_dict(logger, string=f'\n\n** SOURCE TEST RESULT', dict=results)

        # test : target test set
        results = test(model, test_dataloader, unknown_label)
        print_dict(logger, string=f'\n\n** TARGET TEST RESULT', dict=results)
    


    #####################
    #                   #
    #  CALCULATE AUROC  #
    #                   #
    #####################
    logger.info('** CALCULATE AUROC\n\n')
    
    # in-domain predictions
    source_outputs = get_all_predictions(model=model, dataloader=source_test_dataloader, unknown_class=unknown_label)
    source_labels, one_hot_source_labels, source_predictions, source_logits = source_outputs['labels'], source_outputs['one_hot_labels'], source_outputs['predictions'], source_outputs['logits']

    # adaptable-domain predictions
    known_outputs = get_all_predictions(model=model, dataloader=adaptable_dataloader, unknown_class=unknown_label)
    known_labels, one_hot_known_labels, known_predictions, known_logits = known_outputs['labels'], known_outputs['one_hot_labels'], known_outputs['predictions'], known_outputs['logits']

    # unknown-domain predictions
    unknown_outputs = get_all_predictions(model=model, dataloader=unknown_dataloader, unknown_class=unknown_label)
    unknown_labels, one_hot_unknown_labels, unknown_predictions, unknown_logits = unknown_outputs['labels'], unknown_outputs['one_hot_labels'], unknown_outputs['predictions'], unknown_outputs['logits']

    ## in-domain <-> unknown
    labels = np.concatenate([source_labels, unknown_labels])
    predictions = np.concatenate([source_predictions, unknown_predictions])
 
    auroc1 = calculate_auroc(labels, predictions, unknown_index=unknown_label)
    
    ## adaptable <-> unknown
    labels = np.concatenate([known_labels, unknown_labels])
    predictions = np.concatenate([known_predictions, unknown_predictions])
 
    auroc2 = calculate_auroc(labels, predictions, unknown_index=unknown_label)

    if auroc1 < 50 and auroc2 < 50:
        auroc1 = 100 - auroc1
        auroc2 = 100 - auroc2

    
    ## in-domain <-> adaptable
    one_hot_labels = np.concatenate([one_hot_source_labels, one_hot_known_labels])
    logits = np.concatenate([source_logits, known_logits])
    # auroc3 = roc_auc_score(y_true=one_hot_labels, y_score=logits, multi_class='ovo') * 100
    auroc4 = roc_auc_score(y_true=one_hot_labels, y_score=logits, multi_class='ovr') * 100

    logger.info(f'AUROC : IND   <-> UNKNOWN       : {auroc1}')
    logger.info(f'AUROC : ADAPT <-> UNKNOWN       : {auroc2}')
    # logger.info(f'AUROC : IND   <-> ADAPT   (ovo) : {auroc3}')
    logger.info(f'AUROC : IND   <-> ADAPT   (ovr) : {auroc4}')

    logger.info('Done.')


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

