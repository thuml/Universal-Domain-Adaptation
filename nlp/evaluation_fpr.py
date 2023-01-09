
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
from utils.data import get_dataloaders

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

def get_max_logits(model, dataloader):
    max_logits_list = []

    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(tqdm(dataloader, desc='Extracting Max. Logits')):

            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # max_logits  : (batch, )
            # predictions : (batch, )
            max_logits, _ = outputs['max_logits'], outputs['predictions']
            
            max_logits_list.append(max_logits)
    
    max_logits_list = torch.concat(max_logits_list)

    return max_logits_list

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

    
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/{args.method_name}/opda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
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
    train_dataloader, _, eval_dataloader, test_dataloader, source_test_dataloader = get_dataloaders(tokenizer=tokenizer, root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, batch_size=args.test.batch_size, max_length=args.train.max_length)

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

    # cheating test : target test set
    # max_logits_list = get_max_logits(model, test_dataloader)
    max_logits_list = get_max_logits(model, train_dataloader)

    # show results with threshold at 95%
    total_count = len(max_logits_list)
    logger.info(f'Total count : {total_count}')
    sorted_logits, indices = torch.sort(max_logits_list, descending=True)
    logger.info(f'Get  H-score@{args.test.fpr_rate}')
    threshold_index = round(total_count * args.test.fpr_rate)
    threshold = sorted_logits[threshold_index]

    logger.info(f'* H-score @ {args.test.fpr_rate} ...')
    results = test_with_threshold(model, test_dataloader, unknown_label, threshold)
    print_dict(logger, string=f'H-score @ {args.test.fpr_rate} with threshold {threshold}', dict=results)
    

    logger.info('Done.')


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

