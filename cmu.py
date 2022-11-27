
import argparse
import time
import datetime
import logging

import easydict
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Function
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.cmu import CMU
from utils.logging import logger_init, print_dict
from utils.utils import seed_everything
from utils.evaluation import HScore
from utils.data import *

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--lr', type=float, default=None, help='Custom learning rate.')

    args = parser.parse_args()
    lr = args.lr

    config_file = args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    if lr is not None:
        args.train.lr = lr

    return args, save_config

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L226
def norm(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L179
def get_consistency(y_1, y_2, y_3, y_4, y_5):
    y_1 = torch.unsqueeze(y_1, 1)
    y_2 = torch.unsqueeze(y_2, 1)
    y_3 = torch.unsqueeze(y_3, 1)
    y_4 = torch.unsqueeze(y_4, 1)
    y_5 = torch.unsqueeze(y_5, 1)
    c = torch.cat((y_1, y_2, y_3, y_4, y_5), dim=1)
    d = torch.std(c, 1)
    consistency = torch.mean(d, 1)
    return consistency

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L191
def get_entropy(y_1, y_2, y_3, y_4, y_5):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy2 = torch.sum(- y_2 * torch.log(y_2 + 1e-10), dim=1)
    entropy3 = torch.sum(- y_3 * torch.log(y_3 + 1e-10), dim=1)
    entropy4 = torch.sum(- y_4 * torch.log(y_4 + 1e-10), dim=1)
    entropy5 = torch.sum(- y_5 * torch.log(y_5 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))

    entropy = (entropy1 + entropy2 + entropy3 + entropy4 + entropy5) / (5 * entropy_norm)
    return entropy

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L203
def single_entropy(y_1):
    entropy1 = torch.sum(- y_1 * torch.log(y_1 + 1e-10), dim=1)
    entropy_norm = np.log(y_1.size(1))
    entropy = entropy1 / entropy_norm
    return entropy

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L210
def get_confidence(y_1, y_2, y_3, y_4, y_5):
    conf_1, indice_1 = torch.max(y_1, 1)
    conf_2, indice_2 = torch.max(y_2, 1)
    conf_3, indice_3 = torch.max(y_3, 1)
    conf_4, indice_4 = torch.max(y_4, 1)
    conf_5, indice_5 = torch.max(y_5, 1)
    confidence = (conf_1 + conf_2 + conf_3 + conf_4 + conf_5) / 5
    return confidence

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L9
class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:
    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},
    where `i` is the iteration steps.
    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer, init_lr = 0.01, gamma = 0.001, decay_rate = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/model.py#L89
class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None        
        
# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/model.py#L110
class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha = 1.0, lo = 0.0, hi = 1.,
                 max_iters = 1000., auto_step = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/model.py#L137
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct        

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/model.py#L147
class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module, reduction = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, w_s, w_t) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        source_loss = torch.mean(w_s * self.bce(d_s, d_label_s).view(-1))
        target_loss = torch.mean(w_t * self.bce(d_t, d_label_t).view(-1))
        return 0.5 * (source_loss + target_loss)

def train_ensemble(steps_per_epoch, iter, classifier, ensemble, optimizer, scheduler, epoch, index=0):
    # CE-loss
    criterion = nn.CrossEntropyLoss().cuda()

    classifier.eval()
    ensemble.train()

    for ensemble_step in tqdm(range(steps_per_epoch // 2), desc=f'Ensemble {index} at epoch {epoch} '):
        optimizer.zero_grad()
        scheduler.step()

        im, label = next(iter)
        im = im.cuda()
        label = label.cuda()

        with torch.no_grad():
            classifier_prediction, f = classifier(im)
        prediction = ensemble(f.detach(), index)

        loss = criterion(prediction, label)

        # backward
        loss.backward()
        optimizer.step()

# https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/415fbe2fa3a6cb8ef858d991182ccd9ca1ed8960/new/main.py#L306
def evaluate_source_common(classifier, dataloader, ensemble, source_classes, threshold):
    temperature = 1
    classifier.eval()
    ensemble.eval()

    common = []
    target_private = []
    all_confidence = list()
    all_consistency = list()
    all_entropy = list()
    all_labels = list()
    all_output = list()

    source_weight = torch.zeros(len(source_classes)).cuda()
    count = 0
    with torch.no_grad():
        for i, (im, label) in enumerate(dataloader):
            im = im.cuda()

            classifier_prediction, f = classifier(im)
            classifier_prediction = F.softmax(classifier_prediction, -1) / temperature
            ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5 = ensemble(f)

            confidence = get_confidence(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)
            entropy = get_entropy(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)
            consistency = get_consistency(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)

            all_confidence.extend(confidence)
            all_consistency.extend(consistency)
            all_entropy.extend(entropy)

            for prediction in classifier_prediction:
                all_output.append(prediction)

    all_confidence = norm(torch.tensor(all_confidence))
    all_consistency = norm(torch.tensor(all_consistency))
    all_entropy = norm(torch.tensor(all_entropy))

    all_score = (all_confidence + 1 - all_consistency + 1 - all_entropy) / 3

    for i in range(len(all_score)):
        if all_score[i] >= threshold:
            source_weight += all_output[i]
            count +=1

    source_weight = norm(source_weight / count)
    return source_weight

def train(steps_per_epoch, source_iter, target_iter, classifier, domain_discriminator, ensemble, optimizer, 
            lr_scheduler, epoch, source_class_weight):
    
    # CE-loss
    criterion = nn.CrossEntropyLoss().cuda() 
    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discriminator, reduction='none').cuda()

    domain_adv.train()

    classifier.train()
    
    # domain_discriminator.train()
    ensemble.eval()

    for current_step in tqdm(range(steps_per_epoch), desc=f'Train Epoch {epoch}'):
        lr_scheduler.step()
        optimizer.zero_grad()

        # soruce domain
        im_source, label_source = next(source_iter)
        im_source = im_source.cuda()
        label_source = label_source.cuda()
        
        # classifier_prediction_source  : (batch, num_source_class)
        # f_source                      : (btach, 256)
        classifier_prediction_source, f_source = classifier(im_source)
        # classification loss
        classification_loss = criterion(classifier_prediction_source, label_source)

        # target domain
        im_target, _ = next(target_iter)
        im_target = im_target.cuda()
        
        # f_target : (btach, 256)
        classifier_prediction_target, f_target = classifier(im_target)

        with torch.no_grad():
            target_prediction1, target_prediction2, target_prediction3, target_prediction4, target_prediction5 = ensemble(f_target)
            confidence = get_confidence(target_prediction1, target_prediction2, target_prediction3, target_prediction4, target_prediction5)
            entropy = get_entropy(target_prediction1, target_prediction2, target_prediction3, target_prediction4, target_prediction5)
            consistency = get_consistency(target_prediction1, target_prediction2, target_prediction3, target_prediction4, target_prediction5)
            # shape : (batch, )
            w_t = (1 - entropy + 1 - consistency + confidence) / 3
            # shape : (batch, )
            w_s = torch.tensor([source_class_weight[i] for i in label_source]).cuda()


        transfer_loss = domain_adv(f_source, f_target, w_s.detach(), w_t.cuda().detach())
        
        # total loss
        loss = classification_loss + transfer_loss

        # backward
        loss.backward()
        optimizer.step()

    return loss

def test(dataloader, classifier, ensemble, source_classes, unknown_class, threshold):
    metric = HScore(unknown_class)

    classifier.eval()
    ensemble.eval()
   
    all_confidence = list()
    all_consistency = list()
    all_entropy = list()
    all_indices = list()
    all_labels = list()

    with torch.no_grad():
        for i, (im, label) in enumerate(dataloader):
            im = im.cuda()
            label = label.cuda()

            classifier_prediction, f = classifier(im)
            values, indices = torch.max(F.softmax(classifier_prediction, -1), 1)
            ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5 = ensemble(f)

            confidence = get_confidence(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)
            entropy = get_entropy(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)
            consistency = get_consistency(ensemble_prediction1, ensemble_prediction2, ensemble_prediction3, ensemble_prediction4, ensemble_prediction5)

            all_confidence.extend(confidence)
            all_consistency.extend(consistency)
            all_entropy.extend(entropy)
            all_indices.extend(indices)
            all_labels.extend(label)

    all_confidence = norm(torch.tensor(all_confidence))
    all_consistency = norm(torch.tensor(all_consistency))
    all_entropy = norm(torch.tensor(all_entropy))
    all_score = (all_confidence + 1 - all_consistency + 1 - all_entropy) / 3

    for (each_indice, each_label, score) in zip(all_indices, all_labels, all_score):
            if score < threshold:
                prediction = unknown_class
            else:
                prediction = each_indice
            
            predictions = torch.tensor([[prediction]])
            labels = torch.tensor([[each_label]])

            metric.add_batch(predictions=predictions, references=labels)
    
    results = metric.compute()
    return results

    
def main(args, save_config):
    
    ## GPU SETTINGS ##
    # gpu_ids = select_GPUs(args.misc.gpus)
    # TODO : remove?
    gpu_ids = [0]
    output_device = gpu_ids[0]
    ## GPU SETTINGS ##

    ## LOGGINGS ##
    log_dir = f'{args.log.root_dir}/{args.data.dataset.name}/{args.data.dataset.source}-{args.data.dataset.target}/cmu/{args.train.lr}'
    # init logger
    logger_init(logger, log_dir)
    # init tensorboard summarywriter
    if not args.test.test_only:
        writer = SummaryWriter(log_dir)
    # dump configs
    with open(join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    ## LOGGINGS ##


    ## LOAD DATASETS ##
    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)
    source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)

    # dataloaders for ensemble
    dl1, dl2, dl3, dl4, dl5 = esem_dataloader(args, source_classes)

    unknown_class = len(source_classes)
    logger.info(f'Select from {source_classes}, Unknown class {target_private_classes} -> {unknown_class}')
    ## LOAD DATASETS ##


    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = CMU(args, source_classes).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    ## INIT MODEL ##


    ## TEST ONLY ##
    if args.test.test_only:
        logger.info('TEST ONLY...')
        state_dict_path = os.path.join(log_dir, 'best.pth')
        assert os.path.exists(state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))
        results = test(model, target_test_dl, unknown_class)

        print_dict(logger, string='======== Final Test Results ========', dict=results)
        exit(0)
    ## TEST ONLY ##

    # =================== optimizer    
    optimizer = optim.SGD(model.classifier.get_parameters() + model.domain_discriminator.get_parameters(), args.train.lr,
        momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)

    optimizer_ensemble = optim.SGD(model.ensemble.get_parameters() + model.classifier.get_parameters(), args.train.lr, 
        momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)

    lr_scheduler1 = StepwiseLR(optimizer_ensemble, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler2 = StepwiseLR(optimizer_ensemble, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler3 = StepwiseLR(optimizer_ensemble, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler4 = StepwiseLR(optimizer_ensemble, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler5 = StepwiseLR(optimizer_ensemble, init_lr=args.train.lr, gamma=0.001, decay_rate=0.75)

    optimizer_pre = optim.SGD(model.ensemble.get_parameters() + model.classifier.get_parameters(), args.train.lr, 
        momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
    # =================== optimizer


    # total steps / epochs
    steps_per_epoch = max(len(source_train_dl), len(target_train_dl))
    total_epoch = round(args.train.min_step / steps_per_epoch)
    logger.info(f'Total epoch {total_epoch}, steps per epoch {steps_per_epoch}, total step {args.train.min_step}')

    # log every epoch
    log_interval = steps_per_epoch
    # test every epoh
    test_interval = steps_per_epoch

    logger.info(f'Start Training....')
    start_time = time.time()

    source_iter = ForeverDataIterator(source_train_dl)
    target_iter = ForeverDataIterator(source_train_dl)

    # forever iterator for ensemble
    iter1 = ForeverDataIterator(dl1)
    iter2 = ForeverDataIterator(dl2)
    iter3 = ForeverDataIterator(dl3)
    iter4 = ForeverDataIterator(dl4)
    iter5 = ForeverDataIterator(dl5)

    # CE-loss
    criterion = nn.CrossEntropyLoss().cuda()

    # """
    ## Start Pre-training ##
    logger.info('Start pretraining....')
    model.train()
    for pretrain_step in tqdm(range(steps_per_epoch), desc='Pre-training with ensemble'):

        # optimizer zero-grad
        optimizer_pre.zero_grad()

        # ensemble 1
        im, label = next(iter1)
        im = im.cuda()
        label = label.cuda()  
        # classifier_prediction : (batch, num_class)
        # f                     : (batch, hidden_dim)
        # ensemble_prediction   : (batch, num_class)
        classifier_prediction, f, ensemble_prediction = model(im, index=1)
        loss1 = criterion(ensemble_prediction, label)

        # ensemble 2
        im, label = next(iter2)
        im = im.cuda()
        label = label.cuda()  
        classifier_prediction, f, ensemble_prediction = model(im, index=2)
        loss2 = criterion(ensemble_prediction, label)
        
        # ensemble 3
        im, label = next(iter3)
        im = im.cuda()
        label = label.cuda()  
        classifier_prediction, f, ensemble_prediction = model(im, index=3)
        loss3 = criterion(ensemble_prediction, label)
        
        # ensemble 4
        im, label = next(iter4)
        im = im.cuda()
        label = label.cuda()  
        classifier_prediction, f, ensemble_prediction = model(im, index=4)
        loss4 = criterion(ensemble_prediction, label)
        
        # ensemble 5
        im, label = next(iter5)
        im = im.cuda()
        label = label.cuda()  
        classifier_prediction, f, ensemble_prediction = model(im, index=5)
        loss5 = criterion(ensemble_prediction, label)

        # total loss 
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        # backwards
        loss.backward()
        optimizer_pre.step()

    logger.info('Done Pretraining...')
    ## Done Pre-training ##
    # """


    logger.info('Start main training...')

    current_epoch = 0
    best_acc = 0
    best_results = None
    early_stop_count = 0

    ## START TRAINING ##
    for current_epoch in tqdm(range(total_epoch), desc='Train Model'):
        
        ####################
        #                  #
        #       Train      #
        #                  #
        ####################
        # """
        # ensemble 1
        train_ensemble(steps_per_epoch=steps_per_epoch, iter=iter1, classifier=model.classifier, ensemble=model.ensemble, 
            optimizer=optimizer_ensemble, scheduler=lr_scheduler1, epoch=current_epoch, index=1)
    
        # ensemble 2
        train_ensemble(steps_per_epoch=steps_per_epoch, iter=iter2, classifier=model.classifier, ensemble=model.ensemble, 
            optimizer=optimizer_ensemble, scheduler=lr_scheduler2, epoch=current_epoch, index=2)
        
        # ensemble 3
        train_ensemble(steps_per_epoch=steps_per_epoch, iter=iter3, classifier=model.classifier, ensemble=model.ensemble, 
            optimizer=optimizer_ensemble, scheduler=lr_scheduler3, epoch=current_epoch, index=3)
        
        # ensemble 4
        train_ensemble(steps_per_epoch=steps_per_epoch, iter=iter4, classifier=model.classifier, ensemble=model.ensemble, 
            optimizer=optimizer_ensemble, scheduler=lr_scheduler4, epoch=current_epoch, index=4)
        
        # ensemble 5
        train_ensemble(steps_per_epoch=steps_per_epoch, iter=iter5, classifier=model.classifier, ensemble=model.ensemble, 
            optimizer=optimizer_ensemble, scheduler=lr_scheduler5, epoch=current_epoch, index=5)
        # """
        
        source_class_weight = evaluate_source_common(model.classifier, target_test_dl, model.ensemble, source_classes, args.test.threshold)

        # train
        loss = train(steps_per_epoch, source_iter, target_iter, model.classifier, model.domain_discriminator, model.ensemble, optimizer, 
            lr_scheduler, current_epoch, source_class_weight)
    
        ####################
        #                  #
        #     Logging      #
        #                  #
        ####################
        writer.add_scalar('train/loss', loss, current_epoch)


        ####################
        #                  #
        #       Test       #
        #                  #
        ####################

        logger.info(f'TEST at epoch {current_epoch} ...')
        results = test(target_test_dl, model.classifier, model.ensemble, source_classes, unknown_class, args.test.threshold)
        writer.add_scalar('test/mean_acc_test', results['mean_accuracy'], current_epoch)
        writer.add_scalar('test/total_acc_test', results['total_accuracy'], current_epoch)
        writer.add_scalar('test/known_test', results['known_accuracy'], current_epoch)
        writer.add_scalar('test/unknown_test', results['unknown_accuracy'], current_epoch)
        writer.add_scalar('test/hscore_test', results['h_score'], current_epoch)


        if results['mean_accuracy'] > best_acc:
            best_acc = results['mean_accuracy']
            best_results = results
            early_stop_count = 0

            print_dict(logger, string=f'* Best accuracy at epoch {current_epoch}', dict=results)

            logger.info('Saving best model...')
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
            logger.info('Done saving...')
        else:
            print_dict(logger, string=f'* Current accuracy at epoch {current_epoch}', dict=results)

            logger.info('Saving current model...')
            torch.save(model.state_dict(), os.path.join(log_dir, 'current.pth'))
            logger.info('Done saving...')

            if early_stop_count == args.train.early_stop:
                logger.info('End.')
                end_time = time.time()
                logger.info(f'Done training at epoch {current_epoch}. Total time : {end_time-start_time}')     

                print_dict(logger, string=f'** BEST RESULTS', dict=best_results)

                exit()
            early_stop_count += 1
            logger.info(f'Early stopping : {early_stop_count} / {args.train.early_stop}')

    
    print_dict(logger, string=f'** BEST RESULTS', dict=best_results)
    end_time = time.time()
    logger.info(f'Done training full step. Total time : {end_time-start_time}')



if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

