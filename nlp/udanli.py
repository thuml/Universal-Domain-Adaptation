
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
from utils.data import get_datasets, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

ADV_WEIGHT = 0

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
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/udanli-{ADV_WEIGHT}/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
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
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = get_datasets(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class)
    
    #################################
    #                               #
    #  select one samples per class #
    #                               #
    #################################
    

    source_labels_set = sorted(set(train_data[coarse_label]))
    logger.info(f'Select one sample per class : {source_labels_set}')
    selected_samples = dict()
    for source_label in source_labels_set:
        logger.info(f'select label {source_label}')
        # pdb.set_trace()
        filtered_dataset = train_data.filter(lambda sample : sample[coarse_label] == source_label)
        random_index = random.randint(0, len(filtered_dataset)-1)
        selected_sample = filtered_dataset[random_index]
        selected_samples[source_label] = selected_sample

        logger.info(f'SELECTED SAMPLE FOR CLASS {source_label} : {selected_sample}')

    
    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=args.train.max_length, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result

    ## TOKENIZE ##
    # labeled dataset
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    
    # data_collator = default_data_collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # unused in fine-tuning
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=data_collator, batch_size=args.train.batch_size, shuffle=True)
    
    # tokenize train data on-the-fly
    train_dataloader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True)
    eval_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)   
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_data, batch_size=1, shuffle=False) 


    num_step_per_epoch = max(len(train_dataloader), len(train_unlabeled_dataloader))
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

                ## source
                source_batch = next(source_iter)
                # list of input sentences from source domain (batch size)
                source_sentences = source_batch.get(input_key)
                # list of labels (batch_size)
                labels = source_batch.get(coarse_label)

                # entailment
                entailment_batch = []
                for index, label in enumerate(labels):
                    source_sentence = source_sentences[index]
                    entailment_sample = selected_samples.get(label.item()).get(input_key)
                    entailment_batch.append([entailment_sample, source_sentence])

                entailment_batch = tokenizer(entailment_batch, padding=True, return_tensors='pt')
                entailment_batch = {k: v.cuda() for k, v in entailment_batch.items()}

                entailment_output = model(**entailment_batch, is_nli=True)
                entailment_logits = entailment_output.get('logits')

                # contradiction
                contradiction_batch = []
                for index, label in enumerate(labels):
                    source_sentence = source_sentences[index]
                    contradiction_candidates = list(source_labels_set.copy())
                    contradiction_candidates.remove(label)

                    random_label = random.randint(0, len(contradiction_candidates)-1)
                    contradiction_sample = selected_samples.get(random_label).get(input_key)

                    contradiction_batch.append([contradiction_sample, source_sentence])

                contradiction_batch = tokenizer(contradiction_batch, padding=True, return_tensors='pt')
                contradiction_batch = {k: v.cuda() for k, v in contradiction_batch.items()}

                contradiction_output = model(**contradiction_batch, is_nli=True)
                contradiction_logits = contradiction_output.get('logits')

                ## domain adversarial training
                tokenized_source_batch = tokenizer(source_batch.get(input_key), padding=True, return_tensors='pt')
                tokenized_source_batch['labels'] = source_batch.get(coarse_label)
                tokenized_source_batch = {k: v.cuda() for k, v in tokenized_source_batch.items()}
                source_output = model(**tokenized_source_batch, is_nli=False)
                source_domain_logit = source_output.get('domain_output')

                ## target
                target_batch = next(target_iter)
                target_batch = {k: v.cuda() for k, v in target_batch.items()}
                target_output = model(**target_batch, is_nli=False)
                target_domain_logit = target_output.get('domain_output')

                
                ####################
                #                  #
                #   Compute Loss   #
                #                  #
                ####################

                # entailment
                entailment_loss = ce(entailment_logits, torch.full((entailment_logits.shape[0], ), 1).cuda())

                # contradiction
                contradiction_loss = ce(contradiction_logits, torch.full((contradiction_logits.shape[0], ), 0).cuda())

                # domain adversarial
                source_adv_loss = bce(source_domain_logit, torch.ones_like(source_domain_logit))
                target_adv_loss = bce(target_domain_logit, torch.zeros_like(target_domain_logit))

                # total loss
                loss = (1-ADV_WEIGHT) * (entailment_loss + contradiction_loss) + ADV_WEIGHT * (source_adv_loss + target_adv_loss)


                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/entailment_loss', entailment_loss, global_step)
                writer.add_scalar('train/contradiction_loss', contradiction_loss, global_step)
                writer.add_scalar('train/source_adv_loss', source_adv_loss, global_step)
                writer.add_scalar('train/target_adv_loss', target_adv_loss, global_step)

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
            results = eval(model, eval_dataloader, tokenizer, selected_samples, source_labels_set, unknown_label)
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
    results = test(model, test_dataloader, tokenizer, selected_samples, source_labels_set, unknown_label)
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 0)

    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

    logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

