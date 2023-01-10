import os

from torch.utils.data import DataLoader
from transformers import default_data_collator, DataCollatorWithPadding

from datasets import load_dataset

TASK_DICT = {
    'clinc' : 'clinc',
    'trec' : 'trec',
    'massive' : 'AmazonScience/massive',
    'amazon' : 'amazon',
    'visda' : 'visda',
}

# from : https://github.com/thuml/Calibrated-Multiple-Uncertainties/blob/master/new/lib.py#L107
# data iterator that will never stop producing data
class ForeverDataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        
        return data

    def __len__(self):
        return len(self.dataloader)

# load dataset
def load_full_dataset(DATA_PATH, task_name, seed, num_common_class, source=None, target=None):
    # TODO : remove?
    # CHECK CDA SETTING (amazon dataset)
    # amazon-books -> amazon-dvd
    if task_name == 'amazon':
        assert source != None
        assert target != None

        train_data, val_data, source_test_data, test_data, train_unlabeled_data = load_dataset_for_da(source_task=source, target_task=target)
    elif task_name == 'visda':
        pass
    else:
        # UNIDA setting
        # set dataset path
        data_path = os.path.join(DATA_PATH, task_name, f'{seed}_{num_common_class}')
        train_path = os.path.join(data_path, 'source_train.jsonl')
        train_unlabeled_path = os.path.join(data_path, 'target_train_unlabeled.jsonl')
        val_path = os.path.join(data_path, 'source_val.jsonl')
        test_path = os.path.join(data_path, 'target_test.jsonl')
        source_test_path = os.path.join(data_path, 'source_test.jsonl')

        # load dataset
        train_data = load_dataset('json', data_files=train_path)['train']
        train_unlabeled_data = load_dataset('json', data_files=train_unlabeled_path)['train']
        val_data = load_dataset('json', data_files=val_path)['train']
        test_data = load_dataset('json', data_files=test_path)['train']
        source_test_data = load_dataset('json', data_files=source_test_path)['train']
    
    # TODO :
    def add_index(sample, i):
        sample['index'] = i
        return sample

    train_data = train_data.map(add_index, with_indices=True)
    train_unlabeled_data = train_unlabeled_data.map(add_index, with_indices=True)
    val_data = train_data.map(add_index, with_indices=True)
    test_data = test_data.map(add_index, with_indices=True)
    source_test_data = source_test_data.map(add_index, with_indices=True)

    
    return train_data, train_unlabeled_data, val_data, test_data, source_test_data


#########################
#                       #
#       For CDA         #
#                       #
#########################

# for CDA amazon dataset
# returns train, valid, test split
def load_single_dataset(benchmark, task, num_unlabeled):
    
    assert benchmark == 'amazon-multi'
    assert task in ['books', 'dvd', 'electronics', 'kitchen']

    # TODO : customize?
    train_path = f'/home/heyjoonkim/data/datasets/amazon_multi_domain/{task}/train.jsonl'
    test_path = f'/home/heyjoonkim/data/datasets/amazon_multi_domain/{task}/test.jsonl'
    unlabeled_path = f'/home/heyjoonkim/data/datasets/amazon_multi_domain/{task}/unlabeled.jsonl'
    # train_path = f'/home/heyjoonkim/Universal-Domain-Adaptation/data/amazon_multi_domain/{task}/train.jsonl'
    # test_path = f'/home/heyjoonkim/Universal-Domain-Adaptation/data/amazon_multi_domain/{task}/test.jsonl'
    # unlabeled_path = f'/home/heyjoonkim/Universal-Domain-Adaptation/data/amazon_multi_domain/{task}/unlabeled.jsonl'

    train_data = load_dataset('json', data_files=train_path)['train']
    test_data = load_dataset('json', data_files=test_path)['train']
    unlabeled_data = load_dataset('json', data_files=unlabeled_path)['train']

    if num_unlabeled > 0:
        unlabeled_data = unlabeled_data.train_test_split(train_size=num_unlabeled, shuffle=True)['train']

    splits = train_data.train_test_split(test_size=0.2)
    train_data, eval_data = splits['train'], splits['test']

    return train_data, eval_data, test_data, unlabeled_data

# for CDA amazon dataset
# load dataset for DA setting
def load_dataset_for_da(
    source_benchmark='amazon-multi',
    source_task='books',
    target_benchmark='amazon-multi',
    target_task='dvd', 
    num_unlabeled=4000,
):
    
    source_train, source_eval, source_test, source_unlabeled = load_single_dataset(source_benchmark, source_task, num_unlabeled)
    target_train, target_eval, target_test, target_unlabeled = load_single_dataset(target_benchmark, target_task, num_unlabeled)

    return source_train, source_eval, source_test, target_test, target_unlabeled


def get_dataloaders(tokenizer, root_path, task_name, seed, num_common_class, batch_size, max_length, source=None, target=None, drop_last=False):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source, target)
    
    print('# Data per split :')
    print('SOURCE TRAIN / TARGET UNLABELED TRAIN / SOURCE VALIDATION / SOURCE TEST / TARGET TEST')
    print(f'{len(train_data)} / {len(train_unlabeled_data)} / {len(val_data)} / {len(source_test_data)}  / {len(test_data)}')        
    
    # input keys
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

    if source is not None and target is not None:
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        if 'index' in examples:
            result['index'] = examples['index']

        return result

    ## TOKENIZE ##
    # labeled dataset
    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    eval_dataset = val_data.map(
        preprocess_function,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Running tokenizer on source eval dataset",
    )
    test_dataset = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )
    source_test_dataset = source_test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=source_test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )

    # data_collator = default_data_collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    # unused in fine-tuning
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False)   
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    
    return train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader


def get_dataloaders_prompt(tokenizer, root_path, task_name, seed, num_common_class, batch_size, max_length, source=None, target=None, use_soft_prompt=False, use_hard_prompt=False, n_tokens=10):
    ## same as get_dataloaders if use_soft_prompt=False, use_hard_prompt=False
    ## LOAD DATASETS ##
    # source, target, source, target, source
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source, target)
    
    print('# Data per split :')
    print('SOURCE TRAIN / TARGET UNLABELED TRAIN / SOURCE VALIDATION / SOURCE TEST / TARGET TEST')
    print(f'{len(train_data)} / {len(train_unlabeled_data)} / {len(val_data)} / {len(source_test_data)}  / {len(test_data)}')        
    
    # input keys
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

    if source is not None and target is not None:
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'

    # default tokenizing function
    def preprocess_function_base(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result
    def preprocess_function_prompt_soft(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
        for i in range(len(result['input_ids'])):
            ## Add placeholder tokens which are to be replaced with soft tokens later
            result['input_ids'][i] = result['input_ids'][i][:-1] + [500] * n_tokens + [tokenizer.mask_token_id, tokenizer.sep_token_id]
            result['attention_mask'][i] = result['attention_mask'][i][:-1] + [1] * (n_tokens + 2)
            if 'token_type_ids' in result.keys():
                result['token_type_ids'][i] = result['token_type_ids'][i][:-1] + [0] * (n_tokens + 2)
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result
    def preprocess_function_prompt_hard(examples):
        hard_prompt = " It is <mask>"
        hard_prompt_tokenized = tokenizer(hard_prompt, add_special_tokens=False)
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True) # hard prompt ('it is [mask]') length as embedding: 3
        for key in result.keys():
            for i in range(len(result[key])):
                result[key][i] = result[key][i][:-1] + hard_prompt_tokenized[key] + [result[key][i][-1]]
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result
    if use_soft_prompt:        
        preprocess_function = preprocess_function_prompt_soft 
    elif use_hard_prompt:
        preprocess_function = preprocess_function_prompt_hard 
    else:
        preprocess_function = preprocess_function_base
    ## TOKENIZE ##
    # labeled dataset
    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    eval_dataset = val_data.map(
        preprocess_function,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Running tokenizer on source eval dataset",
    )
    test_dataset = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )
    source_test_dataset = source_test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=source_test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )

    # data_collator = default_data_collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True)
    # unused in fine-tuning
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False)   
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    
    return train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader




def get_datasets(root_path, task_name, seed, num_common_class, source=None, target=None):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source, target)
    
    return train_data, train_unlabeled_data, val_data, test_data, source_test_data


def get_nli_datasets(root_path, task_name, seed, num_common_class, num_nli_sample):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class)

    # UNIDA setting
    # set dataset path
    data_path = os.path.join(root_path, task_name, f'{seed}_{num_common_class}')
    nli_path = os.path.join(data_path, f'nli_{num_nli_sample}.jsonl')
    nli_data = load_dataset('json', data_files=nli_path)['train']

    
    return nli_data, train_data, train_unlabeled_data, val_data, test_data, source_test_data


def get_udanli_datasets(root_path, task_name, seed, num_common_class, num_nli_sample, source=None, target=None, is_opda=True):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source=source, target=target)

    # filter out source private samples
    if not is_opda:
        # input keys
        coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
        
        source_label_set = set(train_data[coarse_label])
        target_label_set = set(test_data[coarse_label])
        common_classes = sorted(list(source_label_set.intersection(target_label_set)))
        unknown_class = list(target_label_set - source_label_set)[0]
        new_unknown_class = len(common_classes)

        # mapper : old label -> new label
        label_mapper = dict()
        for new_index, common_class in enumerate(common_classes):
            label_mapper[common_class] = new_index
        label_mapper[unknown_class] = new_unknown_class

        assert len(common_classes) == num_common_class, f'ERROR GENERATING OPDA DATASET : {len(common_classes)} != {num_common_class}'

        print('Filter out source private samples...')
        # ODA : no source private samples, only common class samples
        train_data = train_data.filter(lambda sample : sample[coarse_label] in common_classes)
        val_data = val_data.filter(lambda sample : sample[coarse_label] in common_classes)
        source_test_data = source_test_data.filter(lambda sample : sample[coarse_label] in common_classes)

        print('# Data per split :')
        print('SOURCE TRAIN / TARGET UNLABELED TRAIN / SOURCE VALIDATION / SOURCE TEST / TARGET TEST')
        print(f'{len(train_data)} / {len(train_unlabeled_data)} / {len(val_data)} / {len(source_test_data)}  / {len(test_data)}')        
        

        def update_labels(sample):
            if coarse_label in sample:
                sample[coarse_label] = label_mapper.get(sample[coarse_label])
            return sample
        
        # update label class
        train_data = train_data.map(update_labels)
        val_data = val_data.map(update_labels)
        test_data = test_data.map(update_labels)
        source_test_data = source_test_data.map(update_labels)


    # UNIDA setting
    # set dataset path
    if source is None and target is None:
        data_path = os.path.join(root_path, task_name, f'{seed}_{num_common_class}')
    else:
        data_path = os.path.join(root_path, task_name, f'{source}_{target}', f'{seed}_{num_common_class}')

    filename = f'nli_{num_nli_sample}.jsonl' if is_opda else f'nli_{num_nli_sample}_oda.jsonl'
    nli_path = os.path.join(data_path, filename)

    print(f'Loading NLI data from : {nli_path}')
    nli_data = load_dataset('json', data_files=nli_path)['train']

    
    # UNIDA setting
    # set dataset path
    filename = f'adv_{num_nli_sample}.jsonl' if is_opda else f'adv_{num_nli_sample}_oda.jsonl'
    adv_path = os.path.join(data_path, filename)
    print(f'Loading ADV. data from : {adv_path}')
    adv_data = load_dataset('json', data_files=adv_path)['train']

    
    return nli_data, adv_data, train_data, val_data, test_data, source_test_data



def get_udanli_datasets_v10(root_path, task_name, seed, num_common_class, num_nli_sample, source=None, target=None):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source=source, target=target)

    # UNIDA setting
    # set dataset path
    if source is None and target is None:
        data_path = os.path.join(root_path, task_name, f'{seed}_{num_common_class}')
    else:
        data_path = os.path.join(root_path, task_name, f'{source}_{target}', f'{seed}_{num_common_class}')

    nli_path = os.path.join(data_path, f'nli_{num_nli_sample}.jsonl')

    print(f'Loading NLI data from : {nli_path}')
    nli_data = load_dataset('json', data_files=nli_path)['train']

    
    # UNIDA setting
    # set dataset path
    adv_path = os.path.join(data_path, f'adv_{num_nli_sample}.jsonl')
    print(f'Loading ADV. data from : {adv_path}')
    adv_data = load_dataset('json', data_files=adv_path)['train']

    # UNIDA setting
    # set dataset path
    ent_path = os.path.join(data_path, f'ent_{num_nli_sample}.jsonl')
    print(f'Loading ent. minimization data from : {ent_path}')
    ent_data = load_dataset('json', data_files=ent_path)['train']

    
    return nli_data, adv_data, ent_data, train_data, val_data, test_data, source_test_data


def get_udanli_datasets_v10_1(root_path, task_name, seed, num_common_class, num_nli_sample, source=None, target=None):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source=source, target=target)

    # UNIDA setting
    # set dataset path
    if source is None and target is None:
        data_path = os.path.join(root_path, task_name, f'{seed}_{num_common_class}')
    else:
        data_path = os.path.join(root_path, task_name, f'{source}_{target}', f'{seed}_{num_common_class}')

    nli_path = os.path.join(data_path, f'nli_{num_nli_sample}.jsonl')

    print(f'Loading NLI data from : {nli_path}')
    nli_data = load_dataset('json', data_files=nli_path)['train']

    
    # UNIDA setting
    # set dataset path
    adv_path = os.path.join(data_path, f'adv_{num_nli_sample}.jsonl')
    print(f'Loading ADV. data from : {adv_path}')
    adv_data = load_dataset('json', data_files=adv_path)['train']

    # UNIDA setting
    # set dataset path
    ent_path = os.path.join(data_path, f'ent_{num_nli_sample}-2.jsonl')
    print(f'Loading ent. minimization data from : {ent_path}')
    ent_data = load_dataset('json', data_files=ent_path)['train']

    
    return nli_data, adv_data, ent_data, train_data, val_data, test_data, source_test_data



################
#              #
#   For ODA    #
#              #
################


def get_dataloaders_for_oda(tokenizer, root_path, task_name, seed, num_common_class, batch_size, max_length, source=None, target=None):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source, target)
    
    # input keys
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

    if source is not None and target is not None:
        coarse_label, fine_label, input_key = 'label', 'label', 'sentence'

    
    source_label_set = set(train_data['coarse_label'])
    target_label_set = set(test_data['coarse_label'])
    common_classes = list(source_label_set.intersection(target_label_set))
    unknown_class = list(target_label_set - source_label_set)[0]
    new_unknown_class = num_common_class
    # import pdb
    # pdb.set_trace()
    # assert len(common_classes) == num_common_class, f'ERROR GENERATING OPDA DATASET : {len(common_classes)} != {num_common_class}'

    print('Filter out source private samples...')
    # ODA : no source private samples, only common class samples
    train_data = train_data.filter(lambda sample : sample[coarse_label] in common_classes)
    val_data = val_data.filter(lambda sample : sample[coarse_label] in common_classes)
    source_test_data = source_test_data.filter(lambda sample : sample[coarse_label] in common_classes)


    print('# Data per split :')
    print('SOURCE TRAIN / TARGET UNLABELED TRAIN / SOURCE VALIDATION / SOURCE TEST / TARGET TEST')
    print(f'{len(train_data)} / {len(train_unlabeled_data)} / {len(val_data)} / {len(source_test_data)}  / {len(test_data)}')        
    
    # import pdb
    # pdb.set_trace()

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
        
        if coarse_label in examples:
            # update common class
            for new_label_index, common_class in enumerate(common_classes):
                for i, label in enumerate(examples[coarse_label]):
                    if label == common_class:
                        examples[coarse_label][i] = new_label_index
            # update unknown class
            if unknown_class in examples[coarse_label]:
                for i, label in enumerate(examples[coarse_label]):
                    if label == unknown_class:
                        examples[coarse_label][i] = new_unknown_class

            result["labels"] = examples[coarse_label]

        return result

    ## TOKENIZE ##
    # labeled dataset
    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    train_unlabeled_dataset = train_unlabeled_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_unlabeled_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    eval_dataset = val_data.map(
        preprocess_function,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Running tokenizer on source eval dataset",
    )
    test_dataset = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )
    source_test_dataset = source_test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=source_test_data.column_names,
        desc="Running tokenizer on target test dataset",
    )

    # data_collator = default_data_collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True)
    # unused in fine-tuning
    train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False)   
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    source_test_dataloader = DataLoader(source_test_dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False) 
    
    return train_dataloader, train_unlabeled_dataloader, eval_dataloader, test_dataloader, source_test_dataloader



def get_datasets_for_oda(root_path, task_name, seed, num_common_class, source=None, target=None):
    ## LOAD DATASETS ##
    train_data, train_unlabeled_data, val_data, test_data, source_test_data = load_full_dataset(root_path, task_name, seed, num_common_class, source, target)
    
    # input keys
    coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'
    
    source_label_set = set(train_data[coarse_label])
    target_label_set = set(test_data[coarse_label])
    common_classes = sorted(list(source_label_set.intersection(target_label_set)))
    unknown_class = list(target_label_set - source_label_set)[0]
    # new_unknown_class = len(common_classes)
    new_unknown_class = num_common_class

    # mapper : old label -> new label
    label_mapper = dict()
    for new_index, common_class in enumerate(common_classes):
        label_mapper[common_class] = new_index
    label_mapper[unknown_class] = new_unknown_class

    # assert len(common_classes) == num_common_class, f'ERROR GENERATING OPDA DATASET : {len(common_classes)} != {num_common_class}'

    print('Filter out source private samples...')
    # ODA : no source private samples, only common class samples
    train_data = train_data.filter(lambda sample : sample[coarse_label] in common_classes)
    val_data = val_data.filter(lambda sample : sample[coarse_label] in common_classes)
    source_test_data = source_test_data.filter(lambda sample : sample[coarse_label] in common_classes)

    print('# Data per split :')
    print('SOURCE TRAIN / TARGET UNLABELED TRAIN / SOURCE VALIDATION / SOURCE TEST / TARGET TEST')
    print(f'{len(train_data)} / {len(train_unlabeled_data)} / {len(val_data)} / {len(source_test_data)}  / {len(test_data)}')        
    

    def update_labels(sample):
        if coarse_label in sample:
            sample[coarse_label] = label_mapper.get(sample[coarse_label])
        return sample
    
    # update label class
    train_data = train_data.map(update_labels)
    val_data = val_data.map(update_labels)
    test_data = test_data.map(update_labels)
    source_test_data = source_test_data.map(update_labels)

    return train_data, train_unlabeled_data, val_data, test_data, source_test_data