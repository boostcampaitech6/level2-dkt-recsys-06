import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf

from datetime import datetime
import pytz

import warnings
warnings.filterwarnings("ignore")

logger = get_logger(logging_conf)
korea = pytz.timezone('Asia/Seoul')
current_time = datetime.now(korea).strftime("%m-%d %H:%M")

def main(args):
    wandb.login()

    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    wandb.init(project="level2-dkt", config=vars(args), entity="boostcamp6-recsys6")
    wandb.run.name = "Hyeongjin Cho " + current_time
    wandb.run.save()
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
