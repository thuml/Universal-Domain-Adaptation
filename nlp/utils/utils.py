
import random
import os
import argparse

import easydict
import yaml
import torch
import numpy as np

def seed_everything(seed=1234):
    print(f'SET RANDOM SEED = {seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--lr', type=float, default=None, help='Custom learning rate.')
    parser.add_argument('--threshold', type=float, default=None, help='Custom threshold.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--method_name', type=str, default=None, help='method name to evaluate')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    # for generating dataset
    parser.add_argument('--num_nli_sample', type=int, default=None, help='number of samples for entailment / contradiction')

    args = parser.parse_args()
    lr = args.lr
    seed = args.seed
    method_name = args.method_name
    threshold = args.threshold
    batch_size = args.batch_size
    # for dataset generation
    num_nli_sample = args.num_nli_sample

    config_file = args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    args.train.seed = seed
    if lr is not None:
        args.train.lr = lr
    if method_name is not None:
        args.method_name = method_name
    if threshold is not None:
        args.test.threshold = threshold
    if num_nli_sample is not None:
        args.num_nli_sample = num_nli_sample
    if batch_size is not None:
        args.train.batch_size = batch_size

    return args, save_config