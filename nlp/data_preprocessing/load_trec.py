
import os
import json

from tqdm.auto import tqdm
from datasets import load_dataset




data_path = '/home/heyjoonkim/data/datasets/trec'
split_keys = ['train', 'test']

if not os.path.isdir(data_path):
    os.makedirs(data_path)

# load dataset
trec = load_dataset('trec')

fine_key = 'label-fine'
coarse_key = 'label-coarse'
input_key = 'text'

for split_key in split_keys:
    split_data_dict_list = []
    for sample in tqdm(trec[split_key], desc=f'Loading split {split_key}'):
        data_dict = {
            'text' : sample.get(input_key),
            'coarse_label' : sample.get(coarse_key),
            'fine_label' : sample.get(fine_key),
        }
        split_data_dict_list.append(data_dict)

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