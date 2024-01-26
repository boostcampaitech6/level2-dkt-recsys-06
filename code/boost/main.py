import os
import time

import numpy as np
import pandas as pd
<<<<<<< HEAD

<<<<<<< HEAD
from boost.args import parse_args
from boost.data_loader import Dataset
from boost.utils import get_logger, set_seeds, logging_conf
from boost.trainer import boosting_model

import torch
import wandb
from datetime import datetime
import pytz
=======
from args import parse_args
from data_loader import Dataset, Preprocess
from utils import set_seeds
from trainer import boosting_model
>>>>>>> wooksbaby

import warnings
warnings.filterwarnings("ignore")

# python main.py --model    # CAT, XG, LGBM   default="CAT", 
<<<<<<< HEAD

# Boosting 계열, 수정할수 있는 파라미터
# 1. model 선택, default: CAT, args.py에 있음
# 2. FEATURE 선택 args.py에 있음
# 3. optuna 시도 횟수, default: n_trials=10, 보통 100번이상이면 수렴됨, args.py에 있음
# 4. optuna params, trainer.py에 있음
=======
# python main.py --model CAT --trials 1 --cat_feats userID assessmentItemID testId KnowledgeTag Month DayOfWeek TimeOfDay categorize_solvingTime categorize_ProblemAnswerRate categorize_TagAnswerRate categorize_TestAnswerRate ProblemNumber --n_window 2

# Boosting 계열, 수정할수 있는 파라미터
# 1. FEATURE 선택
# 2. model 선택, default: CAT, args.py에 있음
# 3. train시 valid set 쓸건지 안쓸건지, default: N, args.py에 있음
# 4. optuna 시도 횟수, default: n_trials=50, 보통 100번이상이면 수렴됨, args.py에 있음
# 5. optuna params, trainer.py에 있음
>>>>>>> wooksbaby



def main(args):
<<<<<<< HEAD
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Time
    korea_timezone = pytz.timezone('Asia/Seoul')
    now_korea = datetime.now(korea_timezone)
    now_date = now_korea.strftime('%Y%m%d')
    now_hour = now_korea.strftime('%H%M%S')
    save_time = f"{now_date}_{now_hour}"
    
=======
    ######################## SELECT FEATURE
    # FEATURE = ['userID', 'assessmentItemID', 'testId', 'Timestamp',
    #    'KnowledgeTag', 'SolvingTime', 'CumulativeTime', 'Month', 'DayOfWeek',
    #    'TimeOfDay', 'problems_cumulative', 'problems_last7days',
    #    'problems_last30days', 'CumulativeAnswerRate', 'CumulativeProblemCount',
    #    'ProblemAnswerRate', 'TagAnswerRate', 'TestAnswerRate',
    #    'categorize_solvingTime', 'categorize_ProblemAnswerRate',
    #    'categorize_TagAnswerRate', 'categorize_TestAnswerRate',
    #    'CumulativeUserTagExponentialAverage', 'UserTagCount']

    FEATURE = ['userID', 'assessmentItemID', 'testId',
       'KnowledgeTag', 'SolvingTime', 'CumulativeTime', 'problems_cumulative',
       'problems_last7days', 'problems_last30days',
       'CumulativeProblemCount', 'Month', 'DayOfWeek', 'TimeOfDay',
       'CorrectnessRate', 'TagAccuracy', 'UserCumulativeAnswerRate',
       'categorize_solvingTime', 'categorize_CorrectnessRate',
       'categorize_TagAccuracy']

    # Time
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
>>>>>>> wooksbaby

    ######################## DATA LOAD
    print("### DATA LOAD ###")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
<<<<<<< HEAD
    
    ######################## DATA PREPROCESSING
    print("### DATA PREPROCESSING ###")
    data = Dataset(train, args)
    data = data.split_data()

    ######################## HYPER PARAMETER TUNING - USING OPTUNA
    print("### HYPER PARAMETER TUNING - USING OPTUNA ###")
    print("number of selected features:", len(args.feature))
    model = boosting_model(args, data)

    ######################## TRAIN
    print("### TRAIN ###")
    model.training(data, args)
    
    ######################## INFERENCE
    print("### INFERENCE ###")
  
=======
    #test = pd.read_csv(args.data_dir + args.test_file_name, parse_dates=["Timestamp"])
    #data = Dataset(train, test)

    data = Dataset(train, args)
    data, FE_train = data.split_data(args)

    ######################## DATA PREPROCESSING
    print("### DATA PREPROCESSING ###")
    process = Preprocess(args, data)
    data = process.preprocess()

    ######################## HYPER PARAMETER TUNING - USING OPTUNA
    print("### HYPER PARAMETER TUNING - USING OPTUNA ###")
    print("number of selected features:", len(FEATURE))
    model = boosting_model(args,FEATURE, data)

    ######################## TRAIN
    print("### TRAIN ###")
    model.training(data, args, FEATURE,FE_train)
    
    ######################## INFERENCE
    print("### INFERENCE ###")
>>>>>>> wooksbaby
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data,save_time)

<<<<<<< HEAD
    print(args.model + "_" + save_time + " submission file has been made" )

=======
    print(args.model + " " + save_time + " submission file has been made" )
>>>>>>> wooksbaby
=======
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
>>>>>>> wonhee


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
<<<<<<< HEAD

=======
>>>>>>> wonhee
