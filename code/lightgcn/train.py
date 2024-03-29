import os
import argparse

import torch
import wandb

from lightgcn.args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, set_seeds, logging_conf

from datetime import datetime
import pytz

import warnings

warnings.filterwarnings("ignore")

import pickle

logger = get_logger(logging_conf)
korea = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea).strftime("%m-%d %H:%M")


def main(args: argparse.Namespace):
    wandb.login()
<<<<<<< HEAD
<<<<<<< HEAD
    wandb.init(
        project="level2-lightgcn", config=vars(args), entity="boostcamp6-recsys6"
    )
    wandb.run.name = "Wonhee Lee " + current_time
=======
    wandb.init(project="level2-lightgcn", config=vars(args), entity="boostcamp6-recsys6")
    wandb.run.name = "Yechan Kim " + current_time
>>>>>>> yechan
=======
    wandb.init(project="level2-lightgcn", config=vars(args), entity="boostcamp6-recsys6")
    wandb.run.name = "Hyeongjin Cho " + current_time
>>>>>>> wooksbaby
    wandb.run.save()
    set_seeds(args.seed)

    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
<<<<<<< HEAD
    train_data, test_data, n_node = prepare_dataset(
        device=device, data_dir=args.data_dir
    )
=======
    train_data, test_data, n_node, id2index = prepare_dataset(device=device, data_dir=args.data_dir)
>>>>>>> wooksbaby

    logger.info("Building Model ...")
    model = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers, # default=1
        alpha=args.alpha,
    )
    model = model.to(device)

    logger.info("Start Training ...")
    graph_embedding = trainer.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
    )
    
    # {assessmentid: embedding}
    print(graph_embedding.shape)
    embed_dict = {}
    for id, index in id2index.items():        
        if type(id) == str:
            embed_dict[id] = graph_embedding[index]

    # save graph embedding
    with open(f'../dkt/graph_emb/graph_embed_{current_time}.pickle','wb') as fw:
        pickle.dump(embed_dict, fw)

    wandb.finish()

    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
