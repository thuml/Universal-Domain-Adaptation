
import time
import logging
import copy
import os
import random

import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb

from torch.nn.functional import one_hot
from torch import (
    nn,
    optim
)
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score

from models.udanli import UDANLI
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_udanli_datasets, ForeverDataIterator
from udanli_utils import select_samples

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)


# input keys
coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'


# calculate auroc
def calculate_auroc(labels, predictions, unknown_index):
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=unknown_index)
    roc_auc = auc(fpr, tpr) * 100

    return roc_auc

# get all maximum logits values for calculating auroc
def get_all_predictions(model, dataloader, tokenizer, selected_samples, labels_set, unknown_class):
    one_hot_labels_list = []
    labels_list = []
    predictions_list = []
    logits_list = []
    
    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Testing')):
            eval_sample = test_batch.get(input_key)[0]
            eval_label = test_batch.get(coarse_label).cuda()
            eval_batch = []

            for candidate_label in labels_set:
                candidate_sample = selected_samples.get(candidate_label).get(input_key)
                eval_batch.append([candidate_sample, eval_sample])

            eval_batch = tokenizer(eval_batch, padding=True, return_tensors='pt')
            eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
          
            outputs = model(**eval_batch, is_nli=True)


            total_logits = outputs['logits']
            # predictions = outputs['predictions']

            entailment_logits = total_logits[1, :]
            max_logits = entailment_logits.max(dim=-1).values
            max_index = entailment_logits.max(dim=-1).indices
            best_entailment_logits = total_logits[max_index, 1]

            # for in-domain <-> adaptable-domain (no unknown class)
            # shape : (batch, num_source_class)
            one_hot_label = one_hot(eval_label, num_classes=unknown_class) if unknown_class not in eval_label else None


            labels_list.append(eval_label.cpu().detach().numpy())
            if one_hot_label is not None:
                one_hot_labels_list.append(one_hot_label.cpu().detach().numpy())
            predictions_list.append(max_logits.cpu().detach().numpy())
            logits_list.append(best_entailment_logits.cpu().detach().numpy())

    # pdb.set_trace()

    # shape : (num_samples, )
    # concatenate all predictions and labels
    labels = np.concatenate(labels_list)
    one_hot_labels = np.concatenate(one_hot_labels_list) if len(one_hot_labels_list) > 0 else None
    # predictions = np.concatenate(predictions_list)
    predictions = np.array(predictions_list)
    # logits = np.concatenate(logits_list)
    logits = np.array([logits_list])

    return {
        'labels' : labels,
        'one_hot_labels' : one_hot_labels,
        'predictions' : predictions,
        'logits' : logits,
    }

def test(model, dataloader, tokenizer, selected_samples, labels_set, unknown_class):
    logger.info('Test without threshold.')
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            eval_sample = test_batch.get(input_key)[0]
            eval_label = test_batch.get(coarse_label).cuda()
            eval_batch = []
            for candidate_label in labels_set:
                candidate_sample = selected_samples.get(candidate_label).get(input_key)
                eval_batch.append([candidate_sample, eval_sample])

            eval_batch = tokenizer(eval_batch, padding=True, return_tensors='pt')
            eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
            
            outputs = model(**eval_batch, is_nli=True)

            # entailment = 1
            # predictions : (batch, )
            predictions = outputs['predictions']

            if len(predictions[predictions==1]) == 0:
                predictions = torch.tensor([unknown_class]).cuda()
            elif len(predictions[predictions==1]) == 1:
                predictions = (predictions == 1).nonzero(as_tuple=True)[0].cuda()
            else:
                logits = outputs.get('max_logits')
                logits[predictions != 1] = 0.0
                predictions = logits.argmax().unsqueeze(dim=0).cuda()


            metric.add_batch(predictions=predictions, references=eval_label)
    
    results = metric.compute()

    return results


def main(args, save_config):
    seed_everything(args.train.seed)
    
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/udanli_v9-1/udanli-{args.train.adv_weight}-{args.num_nli_sample}/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
    # init logger
    logger_init(logger, log_dir)

    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')
    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATASETS ##
    nli_data, adv_data, train_data, val_data, test_data, source_test_data = get_udanli_datasets(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, num_nli_sample=args.num_nli_sample, is_opda=False)
    

    source_labels_list = list(sorted(set(train_data[coarse_label])))
        
    # tokenize train data on-the-fly
    eval_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)   
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False) 
    

    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UDANLI(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

    #################################
    #                               #
    #  select one samples per class #
    #                               #
    #################################

    

    # # v9
    # logger.info('Select random samples.')
    # selected_samples = dict()
    # for source_label in source_labels_list:
    #     logger.info(f'select label {source_label}')
    #     # pdb.set_trace()
    #     filtered_dataset = train_data.filter(lambda sample : sample[coarse_label] == source_label)
    #     random_index = random.randint(0, len(filtered_dataset)-1)
    #     selected_sample = filtered_dataset[random_index]
    #     selected_samples[source_label] = selected_sample

    # v9-1
    # dict() : {class_index : sample_instance}
    logger.info('Select samples closest to the center of the distribution.')
    selected_samples = select_samples(
        model=model,
        tokenizer=tokenizer,
        source_labels_list = source_labels_list,
        train_data=train_data,
        coarse_label=coarse_label,
        input_key=input_key,
        batch_size=args.train.batch_size,
    )

    for selected_label, selected_sample in selected_samples.items():
        logger.info(f'SELECTED SAMPLE FOR CLASS {selected_label} : {selected_sample}')


    ####################
    #                  #
    #       Test       #
    #                  #
    ####################

    logger.info('Loading best model ...')
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best.pth')))
            
    logger.info('Test model...')
    results = test(model, test_dataloader, tokenizer, selected_samples, source_labels_list, unknown_label)
    
    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)


    #####################
    #                   #
    #  CALCULATE AUROC  #
    #                   #
    #####################
    logger.info('** CALCULATE AUROC\n\n')

    test_data, source_test_data

    unknown_dataset = test_data.filter(lambda sample: sample[coarse_label] == unknown_label)
    adaptable_dataset = test_data.filter(lambda sample: sample[coarse_label] != unknown_label)

    unknown_dataloader = DataLoader(unknown_dataset, batch_size=1, shuffle=False)   
    adaptable_dataloader = DataLoader(adaptable_dataset, batch_size=1, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_data, batch_size=1, shuffle=False) 

   
    # in-domain predictions
    source_outputs = get_all_predictions(model=model, dataloader=source_test_dataloader, tokenizer=tokenizer, selected_samples=selected_samples, labels_set=source_labels_list, unknown_class=unknown_label)
    source_labels, source_predictions, source_logits = source_outputs['labels'], source_outputs['predictions'], source_outputs['logits']

    # adaptable-domain predictions
    known_outputs = get_all_predictions(model=model, dataloader=adaptable_dataloader, tokenizer=tokenizer, selected_samples=selected_samples, labels_set=source_labels_list, unknown_class=unknown_label)
    known_labels, known_predictions, known_logits = known_outputs['labels'], known_outputs['predictions'], known_outputs['logits']

    # unknown-domain predictions
    unknown_outputs = get_all_predictions(model=model, dataloader=unknown_dataloader, tokenizer=tokenizer, selected_samples=selected_samples, labels_set=source_labels_list, unknown_class=unknown_label)
    unknown_labels, unknown_predictions, unknown_logits = unknown_outputs['labels'], unknown_outputs['predictions'], unknown_outputs['logits']

    ## in-domain <-> unknown
    labels = np.concatenate([source_labels, unknown_labels])
    predictions = np.concatenate([source_predictions, unknown_predictions])
 
    auroc1 = calculate_auroc(labels, predictions, unknown_index=unknown_label)
    
    ## adaptable <-> unknown
    labels = np.concatenate([known_labels, unknown_labels])
    predictions = np.concatenate([known_predictions, unknown_predictions])
 
    auroc2 = calculate_auroc(labels, predictions, unknown_index=unknown_label)

    if auroc1 + auroc2 < 100:
        auroc1 = 100 - auroc1
        auroc2 = 100 - auroc2

    
    # ## in-domain <-> adaptable
    # one_hot_labels = np.concatenate([one_hot_source_labels, one_hot_known_labels])
    # logits = np.concatenate([source_logits, known_logits])
    # auroc3 = roc_auc_score(y_true=one_hot_labels, y_score=logits, multi_class='ovo') * 100
    # auroc4 = roc_auc_score(y_true=one_hot_labels, y_score=logits, multi_class='ovr') * 100

    logger.info(f'AUROC : IND   <-> UNKNOWN       : {auroc1}')
    logger.info(f'AUROC : ADAPT <-> UNKNOWN       : {auroc2}')
    # logger.info(f'AUROC : IND   <-> ADAPT   (ovo) : {auroc3}')
    # logger.info(f'AUROC : IND   <-> ADAPT   (ovr) : {auroc4}')

    logger.info('Done.')



    logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

