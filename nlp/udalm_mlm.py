
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
    DataCollatorForLanguageModeling,
)

from models.udalm import UDALM
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore,Accuracy
from utils.data import get_datasets, ForeverDataIterator
cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)


def main(args, save_config):
    seed_everything(args.train.seed)

    if args.dataset.num_source_class == args.dataset.num_common_class:
        split = 'cda'
    else:
        split = 'opda'

    # amazon reviews data
    if 'source_domain' in args.dataset:
        source_domain = args.dataset.source_domain
        target_domain = args.dataset.target_domain
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/udalm/{source_domain}-{target_domain}/mlm/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
    # clinc, massive, trec
    else:
        source_domain = None
        target_domain = None
        coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
        log_dir = f'{args.log.output_dir}/{args.dataset.name}/udalm/mlm/{split}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
            
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
    _, train_unlabeled_data, _, _, _ = get_datasets(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, source=source_domain, target=target_domain)

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=512, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result

    # TOKENIZE
    # only use unlabeled target domain data for mlm
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    
    # DataLoaders creation :
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.train.mlm_probability)
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)


    num_step_per_epoch = len(train_unlabeled_dataloader)
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
    best_loss = float('inf')

    # data iter
    target_iter = ForeverDataIterator(train_unlabeled_dataloader)

    # CE-loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    
    ## START TRAINING ##
    logger.info(f'Start Training....')
    start_time = time.time()
    for current_epoch in range(1, args.train.num_train_epochs+1):
        model.train()

        for current_step in tqdm(range(num_step_per_epoch), desc=f'TRAIN EPOCH {current_epoch}'):

            global_step += 1
            
            # optimizer zero-grad
            optimizer.zero_grad()

            ####################
            #                  #
            #     Load Data    #
            #                  #
            ####################

            ## target to cuda
            target_batch = next(target_iter)
            target_batch = {k: v.cuda() for k, v in target_batch.items()}
            target_labels = target_batch['labels']

            ####################
            #                  #
            #   Forward Pass   #
            #                  #
            ####################

            target_outputs = model(**target_batch, is_source=False)
            # shape : (batch, length, vocab_size)
            target_logits = target_outputs['logits']

            ####################
            #                  #
            #   Compute Loss   #
            #                  #
            ####################

            loss = ce(target_logits.view(-1, tokenizer.vocab_size), target_labels.view(-1))

            # write to tensorboard
            writer.add_scalar('train/loss', loss, global_step)

            
            # backward, optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        logger.info(f'Save model at epoch {current_epoch}')
        torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))

    
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')
    
    logger.info('Done.')

if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

