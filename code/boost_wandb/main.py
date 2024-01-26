import os
import time

import numpy as np
import pandas as pd
import torch
import wandb

from boost.args import parse_args
from boost.data_loader import Dataset, Preprocess
from boost.utils import get_logger, set_seeds, logging_conf
from boost.boosting import boosting_model

from datetime import datetime
import pytz

import warnings

warnings.filterwarnings("ignore")

# python main.py --model    # CAT, XG, LGBM   default="CAT",

logger = get_logger(logger_conf=logging_conf)
korea = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea).strftime("%m-%d %H:%M")


def main(args):
    wandb.login()

    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ######################## SELECT FEATURE (default: ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"])
    FEATURE = [
        "userID",
        "assessmentItemID",
        "testId",
        "answerCode",
        "KnowledgeTag",
    ] + args.feats

    ######################## DATA LOAD
    logger.info("Loading data ...")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
    test = pd.read_csv(args.data_dir + args.test_file_name, parse_dates=["Timestamp"])

    data = Dataset(train, test)
    data = data.split_data()

    ######################## DATA PREPROCESSING

    logger.info("Preparing data ...")
    process = Preprocess(args, data)
    data = process.preprocess()

    ######################## Hyper parameter tuning
    print("number of selected features:", len(FEATURE))
    
    wandb.init(project="level2-dkt", config=vars(args), entity="boostcamp6-recsys6")
    wandb.run.name = "Wonhee Lee" + current_time
    wandb.run.save()

    logger.info("Building Model ...")
    model = boosting_model(args, FEATURE, data)

    ######################## TRAIN
    logger.info("Start Training ...")
    model.training(data, args, FEATURE)

    ######################## INFERENCE
    logger.info("")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data, current_time)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
