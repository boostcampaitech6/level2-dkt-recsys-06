import os
import time

import numpy as np
import pandas as pd

from boost.args import parse_args
from boost.data_loader import Dataset
from boost.utils import get_logger, set_seeds, logging_conf
from boost.trainer import boosting_model

import torch
import wandb
from datetime import datetime
import pytz

import warnings
warnings.filterwarnings("ignore")

# python main.py --model    # CAT, XG, LGBM   default="CAT", 

# Boosting 계열, 수정할수 있는 파라미터
# 1. model 선택, default: CAT, args.py에 있음
# 2. FEATURE 선택 args.py에 있음
# 3. optuna 시도 횟수, default: n_trials=10, 보통 100번이상이면 수렴됨, args.py에 있음
# 4. optuna params, trainer.py에 있음



def main(args):
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Time
    korea_timezone = pytz.timezone('Asia/Seoul')
    now_korea = datetime.now(korea_timezone)
    now_date = now_korea.strftime('%Y%m%d')
    now_hour = now_korea.strftime('%H%M%S')
    save_time = f"{now_date}_{now_hour}"
    

    ######################## DATA LOAD
    print("### DATA LOAD ###")
    train = pd.read_csv(args.data_dir + args.file_name, parse_dates=["Timestamp"])
    
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
  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.inference(data,save_time)

    print(args.model + "_" + save_time + " submission file has been made" )



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

