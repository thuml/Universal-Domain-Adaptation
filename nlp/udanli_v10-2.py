
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

from torch import (
    nn,
    optim
)
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import (
    AutoTokenizer,
    get_scheduler,
    DataCollatorWithPadding,
)

from models.udanli import UDANLI
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_udanli_datasets_v10_1, ForeverDataIterator
from udanli_utils import select_samples

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

# calculate entropy
# logits : (batch, 2)
def get_entropy(logits):
    # pdb.set_trace()
    entropy = torch.mean(torch.sum(-logits * torch.log(logits + 1e-8), 1), -1)
    return entropy

# input keys
coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

def eval(model, dataloader, tokenizer, selected_samples, labels_set, unknown_class):
    logger.info('Test without threshold.')
    metric = Accuracy()

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

            # predict as unknown
            if len(predictions[predictions==1]) == 0:
                predictions = torch.tensor([unknown_class]).cuda()
            # predict single class
            elif len(predictions[predictions==1]) == 1:
                predictions = (predictions == 1).nonzero(as_tuple=True)[0].cuda()
            # select class with highest logit value
            else:
                logits = outputs.get('max_logits')
                logits[predictions != 1] = 0.0
                predictions = logits.argmax().unsqueeze(dim=0).cuda()

            metric.add_batch(predictions=predictions, references=eval_label)
    
    results = metric.compute()

    return results

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
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/udanli_v10-2/udanli-{args.train.adv_weight}-{args.train.ent_weight}-{args.num_nli_sample}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
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

    ## GET DATASETS ##
    nli_data, adv_data, ent_data, train_data, val_data, test_data, source_test_data = get_udanli_datasets_v10_1(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, num_nli_sample=args.num_nli_sample)
    

    source_labels_list = list(sorted(set(train_data[coarse_label])))
        

    def sentence_pair_preprocess_function(examples):
        sentence1_key = 'text1'
        sentence2_key = 'text2'
        # Tokenize the texts
        texts = (
            (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=args.train.max_length, truncation=True)
 
        if "label" in examples:
            result["labels"] = examples["label"]

        return result

    ## TOKENIZE ##
    # labeled dataset
    nli_dataset = nli_data.map(
        sentence_pair_preprocess_function,
        batched=True,
        remove_columns=nli_data.column_names,
        desc="Running tokenizer on nli dataset",
    )

    adv_dataset = adv_data.map(
        sentence_pair_preprocess_function,
        batched=True,
        remove_columns=adv_data.column_names,
        desc="Running tokenizer on adversarial dataset",
    )

    ent_dataset = ent_data.map(
        sentence_pair_preprocess_function,
        batched=True,
        remove_columns=ent_data.column_names,
        desc="Running tokenizer on entropy minimization dataset",
    )

    # data_collator = default_data_collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # unused in fine-tuning
    nli_dataloader =  DataLoader(nli_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)
    adv_dataloader =  DataLoader(adv_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)
    ent_dataloader =  DataLoader(ent_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)

    # tokenize train data on-the-fly
    eval_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)   
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_data, batch_size=1, shuffle=False) 


    num_step_per_epoch = max(len(adv_dataloader), len(nli_dataloader), len(ent_dataloader))
    total_step = args.train.num_train_epochs * num_step_per_epoch
    logger.info(f'Total epoch {args.train.num_train_epochs}, steps per epoch {num_step_per_epoch}, total step {total_step}')


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UDANLI(
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

    #################################
    #                               #
    #  select one samples per class #
    #                               #
    #################################

    # dict() : {class_index : sample_instance}
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
    best_acc = -1.0
    best_results = None
    early_stop_count = 0


    # data iter
    nli_iter = ForeverDataIterator(nli_dataloader)
    adv_iter = ForeverDataIterator(adv_dataloader)
    ent_iter = ForeverDataIterator(ent_dataloader)

    # cross-entropy loss for classification
    ce = nn.CrossEntropyLoss().cuda()
    # ce = nn.CrossEntropyLoss(reduction='none').cuda()
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
                #   NLI training   #
                #                  #
                ####################

                nli_batch = next(nli_iter)
                nli_batch = {k: v.cuda() for k, v in nli_batch.items()}
                nli_labels = nli_batch['labels']

                nli_output = model(**nli_batch, is_nli=True)
                nli_logits = nli_output.get('logits')

                ## compute loss
                # shape : (batch, )
                nli_loss = ce(nli_logits, nli_labels)

                ####################
                #                  #
                #  Adv. Training   #
                #                  #
                ####################
                
                adv_batch = next(adv_iter)
                adv_batch = {k: v.cuda() for k, v in adv_batch.items()}
                adv_labels = adv_batch['labels']

                adv_output = model(**adv_batch, is_nli=False)
                adv_logit = adv_output.get('domain_output')

                ## compute loss
                adv_loss = bce(adv_logit, adv_labels.unsqueeze(-1).to(torch.float32))

                ####################
                #                  #
                # ent minimization #
                #                  #
                ####################

                ent_batch = next(ent_iter)
                ent_batch = {k: v.cuda() for k, v in ent_batch.items()}

                ent_output = model(**nli_batch, is_nli=True)
                ent_logits = ent_output.get('logits')

                # entropy minimization for target domain (sharp distribution)
                entropy = get_entropy(ent_logits)

                            
                ## total loss
                loss = (1-args.train.adv_weight-args.train.ent_weight) * nli_loss + args.train.adv_weight * adv_loss + args.train.ent_weight * entropy
                # loss = nli_loss + (source_adv_loss + target_adv_loss)



                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/nli_loss', nli_loss, global_step)
                writer.add_scalar('train/adv_loss', adv_loss, global_step)
                writer.add_scalar('train/entropy', entropy, global_step)

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
            results = eval(model, eval_dataloader, tokenizer, selected_samples, source_labels_list, unknown_label)
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
    results = test(model, test_dataloader, tokenizer, selected_samples, source_labels_list, unknown_label)
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 0)

    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

    logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

