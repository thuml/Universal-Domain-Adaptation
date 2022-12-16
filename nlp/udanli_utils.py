

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
    
    # preprocess dataset
    train_data = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Running tokenizer on source train dataset",
    )
    
    # generate distributions for each label
    for source_label in source_labels_list:
        selected_data = train_data.filter(lambda sample : sample["labels"] == source_label)

        dl = DataLoader(selected_data, collate_fn=data_collator, batch_size=batch_size, shuffle=False)
    
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
            centered_embeddings_list = embeddings_list - class_mean

            # calculate variance
            class_variance = LedoitWolf().fit(centered_embeddings_list).precision_.astype(np.float32)

            maha_score_list = []
            for sample_index, embedding in enumerate(embeddings_list):
                

        