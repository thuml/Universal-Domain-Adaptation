
import time
import os
import random
import json

import yaml
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pdb

from tqdm import tqdm

from utils.utils import seed_everything, parse_args
from utils.data import get_datasets, ForeverDataIterator

cudnn.benchmark = True
cudnn.deterministic = True


# input keys
coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'


def main(args, save_config):
    seed_everything(args.train.seed)
    

    source_domain = None
    target_domain = None
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

    data_path = os.path.join(args.dataset.root_path, args.dataset.name, f'{args.train.seed}_{args.dataset.num_common_class}')

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    # pdb.set_trace()

    ## GET DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = get_datasets(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, source=source_domain, target=target_domain)
    
    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_common_class = args.dataset.num_common_class

    source_label_set = set(train_data['coarse_label'])
    target_label_set = set(test_data['coarse_label'])
    common_classes = sorted(list(source_label_set.intersection(target_label_set)))
    unknown_class = list(target_label_set - source_label_set)[0]
    new_unknown_class = len(common_classes)

    assert len(common_classes) == num_common_class, f'ERROR GENERATING OPDA DATASET : {len(common_classes)} != {num_common_class}'

    print('Filter out source private samples...')
    # ODA : no source private samples, only common class samples
    train_data = train_data.filter(lambda sample : sample[coarse_label] in common_classes)
    val_data = val_data.filter(lambda sample : sample[coarse_label] in common_classes)

    

    class2dataset = dict()
    for class_index in common_classes:
        # entailment samples
        entail_dataset_per_class = train_data.filter(lambda sample : sample[coarse_label] == class_index)
        class2dataset[f"{class_index}"] = entail_dataset_per_class
        # contradiction samples
        contradict_dataset_per_class = train_data.filter(lambda sample : sample[coarse_label] != class_index)
        class2dataset[f"-{class_index}"] = contradict_dataset_per_class
        
        # pdb.set_trace()

    for class_index, class_dataset in class2dataset.items():
        print(f'{class_index} > {len(class_dataset)}')

    # TODO : fix?
    NUM_ENTAILMENT = args.num_nli_sample
    NUM_CONTRADICTION = NUM_ENTAILMENT

    
    output_file = os.path.join(data_path, f'nli_{NUM_ENTAILMENT}_oda.jsonl')
    # """
    with open(output_file, 'w') as f:
        for first_sample in tqdm(train_data, desc='Generating NLI samples'):
            label = first_sample.get(coarse_label)

            entail_dataset = class2dataset.get(f"{label}")
            entail_count = len(entail_dataset)

            entail_indices = random.sample(range(entail_count), NUM_ENTAILMENT)

            for entail_index in entail_indices:
                entail_sample = entail_dataset[entail_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : entail_sample.get(input_key),
                    'label' : 1,
                }
                
                # f.write(f'{sample}\n')
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')



            contradiction_dataset = class2dataset.get(f"-{label}")
            contradiction_count = len(contradiction_dataset)

            contradiction_indices = random.sample(range(contradiction_count), NUM_CONTRADICTION)
            
            for contradiction_index in contradiction_indices:
                contradiction_sample = contradiction_dataset[contradiction_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : contradiction_sample.get(input_key),
                    'label' : 0,
                }
                
                # f.write(f'{sample}\n')
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    output_file = os.path.join(data_path, f'adv_{NUM_ENTAILMENT}_oda.jsonl')

    with open(output_file, 'w') as f:
        for first_sample in tqdm(train_data, desc='Generating adversarial samples'):
            
            # source-source sample
            source_count = len(train_data)

            source_indices = random.sample(range(source_count), NUM_ENTAILMENT)

            for source_index in source_indices:
                source_sample = train_data[source_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : source_sample.get(input_key),
                    'label' : 1,
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


            target_count = len(train_unlabeled_data)

            taget_indices = random.sample(range(target_count), NUM_CONTRADICTION)
            
            for target_index in taget_indices:
                target_sample = train_unlabeled_data[target_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : target_sample.get(input_key),
                    'label' : 0,
                }
                
                # f.write(f'{sample}\n')
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    """
    output_file = os.path.join(data_path, f'ent_{NUM_ENTAILMENT}_oda.jsonl')

    with open(output_file, 'w') as f:
        for first_sample in tqdm(train_data, desc='Generating unlabeled target domain samples'):
            
            # source-source sample
            target_count = len(train_unlabeled_data)

            target_indices = random.sample(range(target_count), NUM_ENTAILMENT * 2)

            for target_index in target_indices:
                target_sample = train_unlabeled_data[target_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : target_sample.get(input_key),
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    output_file = os.path.join(data_path, f'ent_{NUM_ENTAILMENT}-2_oda.jsonl')

    with open(output_file, 'w') as f:
        for first_sample in tqdm(train_unlabeled_data, desc='Generating unlabeled target domain samples'):
            
            # source-source sample
            target_count = len(train_unlabeled_data)

            target_indices = random.sample(range(target_count), NUM_ENTAILMENT * 2)

            for target_index in target_indices:
                target_sample = train_unlabeled_data[target_index]

                sample = {
                    'text1' : first_sample.get(input_key),
                    'text2' : target_sample.get(input_key),
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    """

if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

