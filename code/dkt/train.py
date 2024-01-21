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
korea = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea).strftime("%m-%d %H:%M")


def main(args):
    wandb.login()

    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.submission_name = f"{args.model}_current_time"

    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(args=args, file_name=args.file_name)
    data: np.ndarray = preprocess.get_train_data()

    if args.kfolds != 0:
        # folds = preprocess.kfold(data=data, k=args.kfolds)
        folds = preprocess.manual_kfold(data=data, k=args.kfolds)

        for i, (train_data, valid_data) in enumerate(folds):
            wandb.init(
                project="level2-lastquery",
                config=vars(args),
                entity="boostcamp6-recsys6",
            )
            wandb.run.name = f"Wonhee Lee {current_time} {i+1}th fold"
            wandb.run.save()

            print(f"Starting {i+1}th fold ...")
            # train_id, val_id = fold
            # train_data = [data[i] for i in train_id] # sliding window 적용하면 무조건 list 자료형이어야 함 ㅜ
            # valid_data = [data[i] for i in val_id]
            print("data size:", len(train_data), len(valid_data))

            logger.info("Building Model ...")
            model: torch.nn.Module = trainer.get_model(args=args).to(args.device)

            logger.info("Start Training ...")
            args.submission_name = f"{current_time} {i+1}th fold"
            trainer.run(
                args=args, train_data=train_data, valid_data=valid_data, model=model
            )

            wandb.finish()
    else:
        train_data, valid_data = preprocess.split_data(data=data)

        wandb.init(
            project="level2-lastquery", config=vars(args), entity="boostcamp6-recsys6"
        )
        wandb.run.name = "Wonhee Lee " + current_time
        wandb.run.save()

        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)

        logger.info("Start Training ...")
        trainer.run(
            args=args, train_data=train_data, valid_data=valid_data, model=model
        )

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    if args.kfolds != 0 and args.window == True and args.n_choice != 0:
        if args.kfolds % args.n_choice != 0 and args.n_choice % args.kfolds != 0:
            args.error(
                "n_choice는 반드시 kfold의 배수여야 합니다. ex) 5,10 정 안되면 10,5으로 두 명의 유저씩 kfold"
            )

    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
