
import argparse
import time
import datetime
import logging

import easydict
import torch
import yaml
from torch import nn
from tqdm import tqdm

from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pdb

from models.uan import UAN
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
    args = parser.parse_args()

    config_file = args.config

    args = yaml.load(open(config_file))

    save_config = yaml.load(open(config_file))

    args = easydict.EasyDict(args)

    return args, save_config

def main(args, save_config):

    source_classes, target_classes, common_classes, source_private_classes, target_private_classes = get_class_per_split(args)

    for source_index in range(args.data.dataset.num_domains):
        for target_index in range(args.data.dataset.num_domains):
            args.data.dataset['source'] = source_index
            args.data.dataset['target'] = target_index

            print(f'\n\nSOURCE {source_index} -> TARGET {target_index}')

            source_train_dl, source_test_dl, target_train_dl, target_test_dl = get_dataloaders(args, source_classes, target_classes, common_classes, source_private_classes, target_private_classes)


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

