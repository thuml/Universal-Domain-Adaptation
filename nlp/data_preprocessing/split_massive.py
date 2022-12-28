import argparse
import random
import os
import time

from datasets import load_dataset, Dataset
from tqdm.auto import tqdm

DATASET_PATH = '/home/heyjoonkim/data/datasets/massive'
SPLITS = ['train', 'validation', 'test']
FINE_LABEL = 'fine_label'
COARSE_LABEL = 'coarse_label'

SINGLE_DOMAIN_CLASSES = {4, 17}


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset for Universal Domain Adaptation")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )

    parser.add_argument(
        "--num_common_class",
        type=int,
        default=0,
        help="Number of common classes",
    )

    args = parser.parse_args() 

    return args

############################################################
# 
# Train, Validation = Source Private Class + Common Class
# Test              = Target Private Class + Common Class
#
############################################################

def main():
    args = parse_args()

    print(f'RANDOM SEED : {args.seed}')
    random.seed(args.seed)

    output_dir = f'{args.seed}_{args.num_common_class}'
    output_dir = os.path.join(DATASET_PATH, output_dir)
    print(f'SAVE FILTERED DATASET TO : {output_dir}')


    num_common_class = args.num_common_class

    # TRAIN SPLIT
    SPLIT = 'train'
    dataset_path = os.path.join(DATASET_PATH, f'{SPLIT}.jsonl')
    dataset = load_dataset('json', data_files=dataset_path)['train']

    num_fine_labels = len(set(dataset[FINE_LABEL]))
    num_coarse_labels = len(set(dataset[COARSE_LABEL]))
    
    # total class index
    classes = set(range(num_coarse_labels))
    
    assert num_common_class <= num_coarse_labels, f'num_common_class : {num_common_class} > num_coarse_labels : {num_coarse_labels}'
    
    # for CDA setting ONLY
    if num_common_class == 18:
        # remove 2 classes with single subclass
        num_common_class -= 2
        num_fine_labels -= 2
        num_coarse_labels -= 2
        classes = classes - SINGLE_DOMAIN_CLASSES

    
    # select k common classes
    # we remove classes with single domain (in this case class 4 and 17)
    # these classes cannot be splitted into source and target
    while True:
        common_classes = set(random.sample(list(classes), num_common_class))
        if len(SINGLE_DOMAIN_CLASSES.intersection(common_classes)) == 0:
            break

    private_classes = classes - common_classes
    num_private_classes = len(private_classes)
    num_source_private_classes = num_private_classes // 2
    num_target_private_classes = num_private_classes - num_source_private_classes

    assert (num_source_private_classes + num_target_private_classes) == num_private_classes, f'{num_source_private_classes} + {num_target_private_classes} != {num_private_classes}'

    source_private_classes = random.sample(list(private_classes), num_source_private_classes)
    target_private_classes = private_classes - set(source_private_classes)

    print(f'# Common Class         : {num_source_private_classes} : {common_classes}')
    print(f'# Source Private Class : {num_target_private_classes} : {source_private_classes}')
    print(f'# Target Private Class : {num_common_class} : {target_private_classes}')
    print(f'# Total Class          : {num_coarse_labels}')
    print(f'Commonness             : {num_common_class / num_coarse_labels}')
    print('=======================================================\n\n')
    
    source_fine_classes = set()
    target_fine_classes = set()

    # split common class samples
    for common_class in common_classes:
        common_fine_classes = set(dataset.filter(lambda sample: sample[COARSE_LABEL] == common_class)[FINE_LABEL])
        num_common_fine_classes = len(common_fine_classes)
        num_source_common_fine_classes = num_common_fine_classes // 2
        num_target_common_fine_classes = num_common_fine_classes - num_source_common_fine_classes

        source_common_fine_classes = set(random.sample(list(common_fine_classes), num_source_common_fine_classes))
        target_common_fine_classes = common_fine_classes - source_common_fine_classes

        source_fine_classes = source_fine_classes.union(source_common_fine_classes)
        target_fine_classes = target_fine_classes.union(target_common_fine_classes)

    # add source private class samples
    for source_private_class in source_private_classes:
        source_private_fine_classes = set(dataset.filter(lambda sample: sample[COARSE_LABEL] == source_private_class)[FINE_LABEL])
        source_fine_classes = source_fine_classes.union(source_private_fine_classes)

    # add target private class samples
    for target_private_class in target_private_classes:
        target_private_fine_classes = set(dataset.filter(lambda sample: sample[COARSE_LABEL] == target_private_class)[FINE_LABEL])
        target_fine_classes = target_fine_classes.union(target_private_fine_classes)

    # filter selected samples
    source_data = dataset.filter(lambda sample: sample[FINE_LABEL] in source_fine_classes)
    target_data = dataset.filter(lambda sample: sample[FINE_LABEL] in target_fine_classes)

    print('* Filtered TRAIN Stats.')
    print(f'Source ({len(source_data)} samples) : COARSE LABEL CLASS : {len(set(source_data[COARSE_LABEL]))}, FINE LABEL CLASS : {len(set(source_data[FINE_LABEL]))}')
    print(f'Target ({len(target_data)} samples) : COARSE LABEL CLASS : {len(set(target_data[COARSE_LABEL]))}, FINE LABEL CLASS : {len(set(target_data[FINE_LABEL]))}')

    source_coarse_labels = sorted(list(set(source_data[COARSE_LABEL])))
    num_source_coarse_label = len(source_coarse_labels)
    print(num_source_coarse_label, '>', source_coarse_labels)
    label_mapper = dict()
    for new_label in range(num_source_coarse_label):
        source_coarse_label = source_coarse_labels[new_label]
        label_mapper[source_coarse_label] = new_label

    # rename coarse_label    
    coarse_ood_label = num_source_coarse_label

    source_data_dict = source_data.to_dict()
    source_data_dict[COARSE_LABEL] = list(map(lambda coarse_label: label_mapper.get(coarse_label) if coarse_label in source_coarse_labels else coarse_ood_label, source_data_dict[COARSE_LABEL]))
    source_data = Dataset.from_dict(source_data_dict)

    dataset_path = os.path.join(output_dir, f'source_{SPLIT}.jsonl')
    source_data.to_json(dataset_path)

    # remove labels from target data
    target_data = target_data.remove_columns([FINE_LABEL, COARSE_LABEL])

    dataset_path = os.path.join(output_dir, f'target_{SPLIT}_unlabeled.jsonl')
    target_data.to_json(dataset_path)
    ## Done TRAIN split ##



    ## VALIDATION split
    SPLIT = 'validation'
    dataset_path = os.path.join(DATASET_PATH, f'{SPLIT}.jsonl')
    dataset = load_dataset('json', data_files=dataset_path)['train']

    source_data = dataset.filter(lambda sample: sample[FINE_LABEL] in source_fine_classes)

    print('* Filtered VALIDATION Stats.')
    print(f'Source ({len(source_data)} samples) : COARSE LABEL CLASS : {len(set(source_data[COARSE_LABEL]))}, FINE LABEL CLASS : {len(set(source_data[FINE_LABEL]))}')
    
    source_data_dict = source_data.to_dict()
    source_data_dict[COARSE_LABEL] = list(map(lambda coarse_label: label_mapper.get(coarse_label) if coarse_label in source_coarse_labels else coarse_ood_label, source_data_dict[COARSE_LABEL]))
    source_data = Dataset.from_dict(source_data_dict)

    dataset_path = os.path.join(output_dir, f'source_val.jsonl')
    source_data.to_json(dataset_path)
    ## Done VALIDATION split ##



    ## TEST split ##
    SPLIT = 'test'
    dataset_path = os.path.join(DATASET_PATH, f'{SPLIT}.jsonl')
    dataset = load_dataset('json', data_files=dataset_path)['train']

    target_data = dataset.filter(lambda sample: sample[FINE_LABEL] in target_fine_classes)
    soruce_data = dataset.filter(lambda sample: sample[FINE_LABEL] in source_fine_classes)

    print('* Filtered TEST Stats.')
    print(f'Target ({len(target_data)} samples) : COARSE LABEL CLASS : {len(set(target_data[COARSE_LABEL]))}, FINE LABEL CLASS : {len(set(target_data[FINE_LABEL]))}')
    
    # print(f'*** {sorted(list(set(target_data[COARSE_LABEL])))} <-> {source_coarse_labels}')
    # print(label_mapper)

    target_data_dict = target_data.to_dict()
    
    target_data_dict[COARSE_LABEL] = list(map(lambda coarse_label: label_mapper.get(coarse_label) if coarse_label in source_coarse_labels else coarse_ood_label, target_data_dict[COARSE_LABEL]))
    # target_data_dict[COARSE_LABEL] = list(map(tmp, target_data_dict[COARSE_LABEL]))
    target_data = Dataset.from_dict(target_data_dict)

    dataset_path = os.path.join(output_dir, f'target_{SPLIT}.jsonl')
    target_data.to_json(dataset_path)

    # source test data
    source_data_dict = soruce_data.to_dict()
    
    source_data_dict[COARSE_LABEL] = list(map(lambda coarse_label: label_mapper.get(coarse_label) if coarse_label in source_coarse_labels else coarse_ood_label, source_data_dict[COARSE_LABEL]))
    # target_data_dict[COARSE_LABEL] = list(map(tmp, target_data_dict[COARSE_LABEL]))
    soruce_data = Dataset.from_dict(source_data_dict)

    dataset_path = os.path.join(output_dir, f'source_{SPLIT}.jsonl')
    soruce_data.to_json(dataset_path)
    ## Done TEST split ##


    log_path = os.path.join(output_dir, 'stats.txt')
    with open(log_path, 'w') as f:
        f.write(f'# Common Class         : {num_common_class} : {common_classes}\n')
        f.write(f'# Source Private Class : {num_target_private_classes} : {source_private_classes}\n')
        f.write(f'# Target Private Class : {num_source_private_classes} : {target_private_classes}\n')
        f.write(f'# Total Class          : {num_coarse_labels}\n')
        f.write(f'# Commonness           : {num_common_class / num_coarse_labels}\n')
        f.write('=======================================================\n\n')
        f.write(f'# Source fine classes  : {source_fine_classes}\n')
        f.write(f'# Target fine classes  : {target_fine_classes}\n')    
        f.write('=======================================================\n\n')

    print('Done.')


if __name__ == "__main__":
    print('Split MASSIVE.')
    
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total runtime : {end_time - start_time} sec.')