
import os
import json

from tqdm.auto import tqdm
from datasets import load_dataset


data_path = '/home/heyjoonkim/data/datasets/clinc'
domain_file = 'domains.json'
data_file = 'data_full.json'

split_keys = ['train', 'val', 'test']

# load label keys
# domain info.
full_data_path = os.path.join(data_path, domain_file)
with open(full_data_path, 'r') as f:
    domain_keys = json.load(f)

    coarse_labels = list(domain_keys.keys())

coarse_label_count = 0
fine_label_count = 0

coarse_label_mapper = {}
fine_label_mapper = {}
labels_dict = {}

for coarse_label, fine_labels in domain_keys.items():
    if coarse_label not in coarse_label_mapper:
        coarse_label_mapper[coarse_label] = coarse_label_count
        coarse_label_count += 1

    for fine_label in fine_labels:
        labels_dict[fine_label] = coarse_label

        if fine_label not in fine_label_mapper:
            fine_label_mapper[fine_label] = fine_label_count
            fine_label_count += 1

print('* LABELS')
print('coarse label :', len(coarse_labels))
print('fine labels  :', len(labels_dict.keys()))

# load data file
full_data_path = os.path.join(data_path, data_file)
with open(full_data_path, 'r') as f:
    data = json.load(f)

for split_key in split_keys:
    split_data_dict_list = []
    tmp = {}
    for sample in tqdm(data.get(split_key), desc=f'Loading split {split_key}'):
        text, fine_label = sample
        coarse_label = labels_dict.get(fine_label)
        data_dict = {
            'text' : text,
            'coarse_label' : coarse_label_mapper.get(coarse_label),
            'fine_label' : fine_label_mapper.get(fine_label),
        }
        split_data_dict_list.append(data_dict)

        if data_dict['fine_label'] not in tmp:
            tmp[data_dict['fine_label']] = data_dict['coarse_label']

    print(f'SPLIT {split_key} : {len(split_data_dict_list)} samples')

    file_name = f'{split_key}.jsonl'
    file_output_dir = os.path.join(data_path, file_name)
    with open(file_output_dir, 'w') as f:
        for sample in tqdm(split_data_dict_list, desc='writing to output file'):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print('\n\nDone.')


# load saved datasets
for split_key in split_keys:
    full_data_path = os.path.join(data_path, f'{split_key}.jsonl')
    dataset = load_dataset('json', data_files=full_data_path)['train']
    print(f'SPLIT {split_key} : {len(dataset)}')

print('\n\nDone.')