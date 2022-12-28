
import time
import logging
import copy
import os
import random

import torch
import yaml
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pdb

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt  
from tqdm import tqdm
from transformers import AutoTokenizer

from models.udanli import UDANLI
from utils.logging import logger_init
from utils.utils import seed_everything, parse_args
from utils.data import get_udanli_datasets
from udanli_utils import select_samples

cudnn.benchmark = True
cudnn.deterministic = True


logger = logging.getLogger(__name__)

# input keys
coarse_label, fine_label, input_key = 'coarse_label', 'fine_label', 'text'

def main(args, save_config):
    seed_everything(args.train.seed)
    
    ## LOGGINGS ##
    log_dir = f'{args.log.output_dir}/{args.dataset.name}/udanli_v9-1/udanli-{args.train.adv_weight}-{args.num_nli_sample}/opda/common-class-{args.dataset.num_common_class}/{args.train.seed}/{args.train.lr}'
    
    # init logger
    logger_init(logger, log_dir, save_as_file=False)

    # count known class / unknown class
    num_source_labels = args.dataset.num_source_class
    num_class = num_source_labels
    unknown_label = num_source_labels
    logger.info(f'Classify {num_source_labels} + 1 = {num_class+1} classes.\n\n')
    
    ## INIT TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ## GET DATASETS ##
    nli_data, adv_data, train_data, val_data, test_data, source_test_data = get_udanli_datasets(root_path=args.dataset.root_path, task_name=args.dataset.name, seed=args.train.seed, num_common_class=args.dataset.num_common_class, num_nli_sample=args.num_nli_sample)
    
    pdb.set_trace()

    source_labels_list = list(sorted(set(train_data[coarse_label])))
        
    # tokenize train data on-the-fly
    eval_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)   
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False) 
    
    ## INIT MODEL ##
    logger.info('Init model...')
    start_time = time.time()
    model = UDANLI(
        model_name=args.model.model_name_or_path,
        num_class=num_class,
    ).cuda()
    end_time = time.time()
    loading_time = end_time - start_time
    logger.info(f'Done loading model. Total time {loading_time}')
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of trained parameters : {num_total_params}')
    ## INIT MODEL ##

    #################################
    #                               #
    #  select one samples per class #
    #                               #
    #################################

    # v9-1
    # dict() : {class_index : sample_instance}
    logger.info('Select samples closest to the center of the distribution.')
    selected_samples = select_samples(
        model=model,
        tokenizer=tokenizer,
        source_labels_list = source_labels_list,
        train_data=train_data,
        coarse_label=coarse_label,
        input_key=input_key,
        batch_size=args.train.batch_size,
    )

    for selected_label, selected_sample in selected_samples.items():
        logger.info(f'SELECTED SAMPLE FOR CLASS {selected_label} : {selected_sample}')


    ####################
    #                  #
    #       Test       #
    #                  #
    ####################

    model_path = os.path.join(log_dir, 'best.pth')
    logger.info(f'Loading best model from : {model_path}')
    model.load_state_dict(torch.load(model_path))
            
    logger.info('Extract Embeddings...')

    all_embeddings = []
    all_plot_labels = []
    for test_sample in tqdm(test_dataloader, desc='Extracting embeddings'):
        test_sample_label = test_sample[coarse_label]
        test_sample_sentence = test_sample.get(input_key)[0]

        batch = []
        for inference_label, inference_sample in selected_samples.items():
            plot_label = inference_label if inference_label == test_sample_label else unknown_label
            inference_sample_sentence = inference_sample.get(input_key)
            batch.append([inference_sample_sentence, test_sample_sentence])
            # shape : (embedding_dim, )
            all_plot_labels.append(plot_label)
        # pdb.set_trace()
        eval_batch = tokenizer(batch, padding=True, return_tensors='pt')
        eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
          
        # shape : (batch, embedding_dim)
        embeddings = model(**eval_batch, embeddings_only=True)
        embeddings = embeddings.cpu().detach().numpy()
        all_embeddings.append(embeddings)

    # shape : (num_samples, embedding_dim)
    all_embeddings = np.concatenate(all_embeddings)

    fig = plt.figure(figsize=(6,6))
    logger.info('Train T-SNE....')
    tsne = TSNE(random_state=args.train.seed, n_iter=5000)

    rep_tsne = tsne.fit_transform(all_embeddings)

    color_list = ['red', 'olive', 'lawngreen', 'aqua', 'dodgerblue', 'blue', 'indigo', 'black']

    for i in range(len(all_embeddings)): 
        plt.scatter(rep_tsne[i, 0], rep_tsne[i, 1], marker='o', color=color_list[all_plot_labels[i]], s=12, linewidths=0.5)
    


    ax = plt.gca()
    # ax.set_size_inches(8, 8)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


    plt.xlim(rep_tsne[:, 0].min()-5, rep_tsne[:, 0].max()+5) # 최소, 최대
    plt.ylim(rep_tsne[:, 1].min()-5, rep_tsne[:, 1].max()+5) # 최소, 최대
    
    plt.savefig(f'tsne_demon.png',bbox_inches='tight', dpi=1000)
    plt.close('all')

    logger.info('Done.')    


if __name__ == "__main__":
        
    args, save_config = parse_args()
    main(args, save_config)

