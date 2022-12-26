
import time
import logging
import copy
import os

import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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

from models.ovanet import OVANET
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything, parse_args
from utils.evaluation import HScore, Accuracy
from utils.data import get_dataloaders_for_oda, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

def eval(model, dataloader):
    logger.info('Test without threshold.')
    metric = Accuracy()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # predictions : (batch, )
            predictions = outputs['predictions']

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()

    return results

def test(model, dataloader, unknown_class):
    logger.info('Test without threshold.')
    metric = HScore(unknown_class)

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(dataloader, desc='testing '):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            labels = test_batch['labels']

            outputs = model(**test_batch)

            # predictions : (batch, )
            predictions = outputs['predictions']

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()

    return results

# from original code
# https://github.com/VisionLearningGroup/OVANet/blob/d40020d2d59e617ca693ce5195b7b5a44a9893d5/utils/loss.py#L14
def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.range(0, out_open.size(0) - 1).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

# from original code
# https://github.com/VisionLearningGroup/OVANet/blob/d40020d2d59e617ca693ce5195b7b5a44a9893d5/utils/loss.py#L30
def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open


def main(args, save_config):
    seed_everything(args.train.seed)
    
    
    source_domain = None
    target_domain = None
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/ovanet/oda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'


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
    model = OVANET(
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
                model.C2.weight_norm()

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
                # shape : (batch, num_class)
                source_logits, source_logits_open = source_outputs['logits'], source_outputs['logits_open']
                
                ## target ##
                target_outputs = model(**target_batch)
                # shape : (batch, num_class)
                target_logits, target_logits_open = target_outputs['logits'], target_outputs['logits_open']
                
                ####################
                #                  #
                #   Compute Loss   #
                #                  #
                ####################

                ## source ##
                ce_loss = ce(source_logits, source_labels)

                # shape : (batch, 2, num_souce_class)
                source_logits_open = source_logits_open.view(source_logits_open.size(0), 2, -1)
                open_loss_pos, open_loss_neg = ova_loss(source_logits_open, source_labels)
                open_loss_source = 0.5 * (open_loss_pos + open_loss_neg)

                ## target ##
                # shape : (batch, 2, num_souce_class)
                target_logits_open = target_logits_open.view(target_logits_open.size(0), 2, -1)
                ent_open = open_entropy(target_logits_open)
                ent_loss_target = args.train.multi * ent_open

                # total loss
                loss = ce_loss + open_loss_source + ent_loss_target

                # write to tensorboard
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/ce_loss', ce_loss, global_step)
                writer.add_scalar('train/open_loss', open_loss_source, global_step)
                writer.add_scalar('train/ent_loss', ent_loss_target, global_step)

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
            results = eval(model, eval_dataloader)
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
    results = test(model, test_dataloader, unknown_label)
    for k,v in results.items():
        writer.add_scalar(f'test/{k}', v, 0)

    print_dict(logger, string=f'\n\n** FINAL TARGET DOMAIN TEST RESULT', dict=results)

    logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

