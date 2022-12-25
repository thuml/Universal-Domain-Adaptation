import pdb

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding
from sklearn.covariance import LedoitWolf


def select_samples(model, tokenizer, source_labels_list, train_data, coarse_label, input_key, batch_size=64):
    data_collator = DataCollatorWithPadding(tokenizer)

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result
    
    # # preprocess dataset
    # train_data = train_data.map(
    #     preprocess_function,
    #     batched=True,
    #     remove_columns=train_data.column_names,
    #     desc="Running tokenizer on source train dataset",
    # )
    
    selected_samples = dict()
    # generate distributions for each label
    for source_label in source_labels_list:
        selected_data = train_data.filter(lambda sample : sample[coarse_label] == source_label)

        tokenized_selected_data = selected_data.map(
            preprocess_function,
            batched=True,
            remove_columns=selected_data.column_names,
            desc="Running tokenizer on source train dataset",
        )

        dl = DataLoader(tokenized_selected_data, collate_fn=data_collator, batch_size=batch_size, shuffle=False)
    
        with torch.no_grad():
            embeddings_list = []
            for batch in dl:
                batch = {k: v.cuda() for k, v in batch.items()}

                # shape : (batch, hidden_dim)
                embeddings = model(**batch, embeddings_only=True)

                embeddings_list.append(embeddings)

            # shape : (num_samples, hidden_dim)
            embeddings_list = torch.concat(embeddings_list)

            # shape : (hidden_dim, )
            class_mean = embeddings_list.mean(0)

            # shape : (num_samples, hidden_dim)
            centered_embeddings_list = (embeddings_list - class_mean).clone().detach().cpu().numpy()

            # calculate variance
            # shape : (hidden_dim, hidden_dim)
            class_variance = LedoitWolf().fit(centered_embeddings_list).precision_.astype(np.float32)
            class_variance = torch.from_numpy(class_variance).float().cuda()

            # shape : (num_samples, hidden_dim)
            centered_embeddings_list = embeddings_list - class_mean
            # calculate mahalanobis distance
            # shape : (num_samples, )
            maha_scores = torch.diag(centered_embeddings_list @ class_variance @ centered_embeddings_list.t())
            
            selected_index = maha_scores.argmin().item()

            selected_sample = selected_data[selected_index]

            selected_samples[source_label] = selected_sample
                
    # pdb.set_trace()

    return selected_samples
        




def select_samples_v2(model, tokenizer, source_labels_list, train_data, coarse_label, input_key, batch_size=64):
    data_collator = DataCollatorWithPadding(tokenizer)

    # default tokenizing function
    def preprocess_function(examples):
        texts = (examples[input_key],)
        result = tokenizer(*texts, padding=False, truncation=True)
        
        if coarse_label in examples:
            result["labels"] = examples[coarse_label]

        return result
    
    
    selected_samples = dict()
    # generate distributions for each label
    for source_label in source_labels_list:
        selected_data = train_data.filter(lambda sample : sample[coarse_label] == source_label)

        tokenized_selected_data = selected_data.map(
            preprocess_function,
            batched=True,
            remove_columns=selected_data.column_names,
            desc="Running tokenizer on source train dataset",
        )

        dl = DataLoader(tokenized_selected_data, collate_fn=data_collator, batch_size=batch_size, shuffle=False)
    
        with torch.no_grad():
            embeddings_list = []
            labels_list = []
            for batch in dl:
                batch = {k: v.cuda() for k, v in batch.items()}

                # shape : (batch, hidden_dim)
                embeddings = model(**batch, embeddings_only=True)

                embeddings_list.append(embeddings)
                labels_list.append(batch['labels'])

            # shape : (num_samples, hidden_dim)
            embeddings_list = torch.concat(embeddings_list)
            # shape : (num_samples, hidden_dim)
            norm_embeddings_list = F.normalize(embeddings_list, dim=-1)
            # shape : (num_samples, )
            labels_list = torch.concat(labels_list)

            # shape : (hidden_dim, )
            class_mean = norm_embeddings_list.mean(0)

            # shape : (num_samples, )
            cosine_similarities = class_mean @ norm_embeddings_list.t()

            best_cosine_score, best_index = cosine_similarities.max(-1)

            # pdb.set_trace()

            selected_sample = selected_data[best_index.item()]

            selected_samples[source_label] = selected_sample
                
    # pdb.set_trace()

    return selected_samples
        