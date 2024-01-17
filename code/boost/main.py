import os
import time

import numpy as np
import pandas as pd

from args import parse_args
from data_loader import Dataset, Preprocess
from utils import set_seeds
from trainer import boosting_model

import warnings
warnings.filterwarnings("ignore")

# python main.py --model    # CAT, XG, LGBM   default="CAT", 

# Boosting 계열, 수정할수 있는 파라미터
# 1. FEATURE 선택
# 2. model 선택, default: CAT, args.py에 있음
# 3. train시 valid set 쓸건지 안쓸건지, default: N, args.py에 있음
# 4. optuna 시도 횟수, default: n_trials=10, 보통 100번이상이면 수렴됨, args.py에 있음
# 5. optuna params, trainer.py에 있음



def main(args):
    ######################## SELECT FEATURE
    FEATURE = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 
               'SolvingTime', 
               'CumulativeTime', 'problems_cumulative', 'problems_last7days', 'problems_last30days', 'Month', 'DayOfWeek', 'TimeOfDay', 
               'categorize_solvingTime', 
               'categorize_CorrectnessRate', 
               'categorize_TagAccuracy'
    ]

    # Time
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')

    ######################## DATA LOAD
    print("### DATA LOAD ###")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
    test = pd.read_csv(args.data_dir + args.test_file_name, parse_dates=["Timestamp"])

    data = Dataset(train, test)
    data, FE_train = data.split_data()

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data,save_time)

    print(args.model + " " + save_time + " submission file has been made" )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

