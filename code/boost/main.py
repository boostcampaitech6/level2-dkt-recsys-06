import os
import time

import numpy as np
import pandas as pd

from args import parse_args
from data_loader import Dataset, Preprocess
from utils import set_seeds
from boosting import boosting_model

import warnings
warnings.filterwarnings("ignore")

import optuna

# python main.py --model    # CAT, XG, LGBM   default="CAT", 


def main(args):
    ######################## SELECT FEATURE
    FEATURE = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 
               'SolvingTime', 
               'CumulativeTime', 'problems_cumulative', 'problems_last7days', 'problems_last30days', 'Month', 'DayOfWeek', 'TimeOfDay', 
               'categorize_solvingTime', 
               'categorize_CorrectnessRate', 
               'categorize_TagAccuracy'
    ]
    
    ######################## DATA LOAD
    # Time
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')

    print("Load Data")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
    test = pd.read_csv(args.data_dir + args.test_file_name, parse_dates=["Timestamp"])

    data = Dataset(train, test)
    data = data.split_data()

    ######################## DATA PREPROCESSING

    print("Preprocessing Data")
    process = Preprocess(args, data)
    data = process.preprocess()

    ######################## Hyper parameter tuning
    print("number of selected features:", len(FEATURE))
    print("Hyper parameter tuning:")
    model = boosting_model(args,FEATURE, data)

    ######################## TRAIN
   
    model.training(data, args, FEATURE)

    ######################## INFERENCE

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data,save_time)
    print(save_time)



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

