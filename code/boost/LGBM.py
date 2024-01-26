import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import os
import random

import yaml
from datetime import datetime
import pytz

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import wandb
from wandb.lightgbm import wandb_callback, log_summary
from typing import TYPE_CHECKING, Callable
from wandb.sdk.lib import telemetry as wb_telemetry
from typing import List
from lightgbm.callback import CallbackEnv

MINIMIZE_METRICS = [
    "l1",
    "l2",
    "rmse",
    "mape",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "binary_logloss",
]

MAXIMIZE_METRICS = ["map", "auc", "average_precision"]

def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def wandb_callback(log_params=True, define_metric=True) -> Callable:
    """Automatically integrates LightGBM with wandb.

    Arguments:
        log_params: (boolean) if True (default) logs params passed to lightgbm.train as W&B config
        define_metric: (boolean) if True (default) capture model performance at the best step, instead of the last step, of training in your `wandb.summary`

    Passing `wandb_callback` to LightGBM will:
      - log params passed to lightgbm.train as W&B config (default).
      - log evaluation metrics collected by LightGBM, such as rmse, accuracy etc to Weights & Biases
      - Capture the best metric in `wandb.summary` when `define_metric=True` (default).

    Use `log_summary` as an extension of this callback.

    Example:
        ```python
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            .
        }
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10,
                        valid_sets=lgb_eval,
                        valid_names=('validation'),
                        callbacks=[wandb_callback()])
        ```
    """
    def _define_metric(data: str, metric_name: str) -> None:
    
        """Capture model performance at the best step.
        instead of the last step, of training in your `wandb.summary`
        """
        if "loss" in str.lower(metric_name):
            wandb.define_metric(f"{data}_{metric_name}", summary="min")
        elif str.lower(metric_name) in MINIMIZE_METRICS:
            wandb.define_metric(f"{data}_{metric_name}", summary="min")
        elif str.lower(metric_name) in MAXIMIZE_METRICS:
            wandb.define_metric(f"{data}_{metric_name}", summary="max")
            
    log_params_list: "List[bool]" = [log_params]
    define_metric_list: "List[bool]" = [define_metric]

    def _init(env: "CallbackEnv") -> None:
        with wb_telemetry.context() as tel:
            tel.feature.lightgbm_wandb_callback = True

        wandb.config.update(env.params)
        log_params_list[0] = False

        if define_metric_list[0]:
            for i in range(len(env.evaluation_result_list)):
                data_type = env.evaluation_result_list[i][0]
                metric_name = env.evaluation_result_list[i][1]
                _define_metric(data_type, metric_name)

    def _callback(env: "CallbackEnv") -> None:
        if log_params_list[0]:
            _init(env)
        # eval_results: "Dict[str, Dict[str, List[Any]]]" = {}
        # recorder = lightgbm.record_evaluation(eval_results)
        # recorder(env)
        eval_results = {x[0]:{x[1:][0]:x[1:][1:]} for x in env.evaluation_result_list}

        for validation_key in eval_results.keys():
            for key in eval_results[validation_key].keys():
                 wandb.log(
                     {validation_key + "_" + key: eval_results[validation_key][key][0]},
                     commit=False,
                 )
        for item in eval_results:
            if len(item) == 4:
                wandb.log({f"{item[0]}_{item[1]}": item[2]}, commit=False)

        # Previous log statements use commit=False. This commits them.
        wandb.log({"iteration": env.iteration}, commit=True)

    return _callback

sweep_config_path = '/data/ephemeral/home/level2-dkt-recsys-06/code/boost/lgbmsweepconfig.yaml'

# 노트북의 이름 설정
os.environ['WANDB_NOTEBOOK_NAME'] = 'LGBM.py'

# YAML 파일 로드
with open(sweep_config_path, 'r') as file:
    sweep_config = yaml.safe_load(file)

# W&B 스위프트 설정
sweep_id = wandb.sweep(sweep=sweep_config, project="lightgbm-sweep")

# 시드 고정
set_seeds()



Feature = [ 'Itemseq', 'SolvingTime', 'CumulativeTime', 'UserAvgSolvingTime',
       'RelativeUserAvgSolvingTime', 'CumulativeItemCount', 'Item_last7days',
       'Item_last30days', 'CumulativeUserItemAcc', 'PastItemCount',
       'UserItemElapsed', 'UserRecentItemSolvingTime', 'ItemAcc',
       'AverageItemSolvingTime_Correct', 'AverageItemSolvingTime_Incorrect',
       'AverageItemSolvingTime', 'RelativeItemSolvingTime',
       'SolvingTimeClosenessDegree', 'UserTagAvgSolvingTime', 'TagAcc',
       'CumulativeUserTagAverageAcc', 'CumulativeUserTagExponentialAverage',
       'UserTagCount', 'UserTagElapsed',  'TestAcc', ]

Categorical_Feature = ['userID', 'assessmentItemID', 'testId','KnowledgeTag',
                       'Month','DayOfWeek', 'TimeOfDay', 'WeekOfYear', 
       'UserRecentTagAnswer',
       'UserRecentItemAnswer',
    #   'categorize_solvingTime',
    #   'categorize_ItemAcc', 'categorize_TagAcc', 'categorize_TestAcc',
    #   'categorize_CumulativeUserItemAcc',
    #   'categorize_CumulativeUserTagAverageAcc',
    #   'categorize_CumulativeUserTagExponentialAverage', 'CategorizedDegree'

]
Feature = Feature + Categorical_Feature


## python LGBM.py


data=pd.read_csv('/data/ephemeral/home/level2-dkt-recsys-06/data/FE_v9.csv')
data = data.sort_values(by=["userID", "Timestamp", "assessmentItemID"]).reset_index(drop=True)
test = data[data["answerCode"] == -1]
X = data[data['answerCode']!=-1]

col=Feature+['answerCode']


test=test[col]
X=X[col]

# as category: integer여도 범주형으로 취급 가능
for feature in Categorical_Feature:
       test[feature] = test[feature].astype('category')
       X[feature] = X[feature].astype('category')



print(X.dtypes)




default_config = {
    "num_leaves": 10,  # 최소값 10
    "learning_rate": 0.0001,  # 최소값 0.0001
    "max_depth": -1,  # -1 (깊이 제한 없음)
    "min_data_in_leaf": 20,  # 최소값 20
    "feature_fraction": 0.6,  # 최소값 0.6
    "bagging_fraction": 0.6,  # 최소값 0.6
    "bagging_freq": 0,  # 최소값 0
    "lambda_l1": 0.0,  # 최소값 0.0
    "lambda_l2": 0.0,  # 최소값 0.0
    "cat_smooth": 10,  # 최소값 10
}

def train():
    
    auc = 0
    acc = 0
    test_preds = np.zeros(len(test))
    
    wandb.init(project= "lightgbm-sweep", config=default_config,entity= "boostcamp6-recsys6")
    
    ratio = wandb.config.ratio
    
    sampled_indices = X.groupby('userID').sample(frac=ratio).index

    # userID별 마지막 인덱스 찾기
    # last_indices = X.groupby("userID").tail(1).index

    # 학습 데이터셋 생성
    X_train = X.drop(sampled_indices)
    y_train = X_train["answerCode"]

    # 검증 데이터셋 생성
    X_valid = X.loc[sampled_indices]
    y_valid = X_valid["answerCode"]

    print(X_train.dtypes)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    
    # 완드비 실험 이름
    korea = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea).strftime("%m-%d %H:%M")
    wandb.run.name = f"yechan {current_time}"
    current_params = {
        "objective": "binary",
        "metric": ["auc"],
        "device": "cpu",
        "num_leaves": wandb.config.num_leaves,
        "learning_rate": wandb.config.learning_rate,
        "max_depth": wandb.config.max_depth,
        "min_data_in_leaf": wandb.config.min_data_in_leaf,
        "feature_fraction": wandb.config.feature_fraction,
        "bagging_fraction": wandb.config.bagging_fraction,
        "bagging_freq": wandb.config.bagging_freq,
        "lambda_l1": wandb.config.lambda_l1,
        "lambda_l2": wandb.config.lambda_l2,
        "cat_smooth": wandb.config.cat_smooth,
    }
    model = lgb.train(
        current_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=500,
        callbacks=[
            wandb_callback(log_params=True, define_metric=True),
            lgb.early_stopping(30),
        ],
        categorical_feature=[
            "userID",
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "Month"
        ],
    )
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)
    test_preds += model.predict(test)
    print(f"VALID AUC : {auc} ACC : {acc}\n")
    wandb.log({"auc": auc, "accuracy": acc})
    wandb.finish()
    
    #output파일 생성
    output_dir = "output/"
    write_path = os.path.join(
        output_dir,
        f"auc:{auc} acc:{acc} lgbm{current_time}.csv",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(test_preds):
            w.write("{},{}\n".format(id, p))
            
    feature_importances = model.feature_importance()
    feature_names = model.feature_name()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

    print(importance_df)


wandb.agent(sweep_id, train)



