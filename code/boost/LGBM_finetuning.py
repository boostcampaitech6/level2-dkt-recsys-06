import warnings 
warnings.filterwarnings("ignore")

import pandas as pd
import os
import random
import numpy as np
import yaml
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pytz
from datetime import datetime, timezone, timedelta
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from wandb.lightgbm import wandb_callback, log_summary

import optuna
import json

X = pd.read_csv('/data/ephemeral/home/level2-dkt-recsys-06/data/FE_v9.csv')
# test =  pd.read_csv('/data/ephemeral/home/level2-dkt-recsys-06/data/FE_v9_test.csv')
X = X.sort_values(by=["userID", "Timestamp", "assessmentItemID"]).reset_index(drop=True)
# test = test.sort_values(by=["userID", "Timestamp", "assessmentItemID"]).reset_index(drop=True)

test = X[X["answerCode"] == -1]
X = X[X['answerCode']!=-1]

Feature = [ 'Itemseq', 'SolvingTime', 'CumulativeTime', 'UserAvgSolvingTime',
       'RelativeUserAvgSolvingTime', 'CumulativeItemCount', 'Item_last7days',
       'Item_last30days', 'CumulativeUserItemAcc', 'PastItemCount',
       'UserItemElapsed', 'UserRecentItemSolvingTime', 'ItemAcc',
       'AverageItemSolvingTime_Correct', 'AverageItemSolvingTime_Incorrect',
       'AverageItemSolvingTime', 'RelativeItemSolvingTime',
       'SolvingTimeClosenessDegree', 'UserTagAvgSolvingTime', 'TagAcc',
       'CumulativeUserTagAverageAcc', 'CumulativeUserTagExponentialAverage',
       'UserTagCount', 'UserTagElapsed',  'TestAcc']

Categorical_Feature = ['userID', 'assessmentItemID', 'testId','KnowledgeTag',
                       'Month','DayOfWeek', 'TimeOfDay', 'WeekOfYear', 
       'UserRecentTagAnswer',
       'UserRecentItemAnswer',
       'categorize_solvingTime',
       'categorize_ItemAcc', 'categorize_TagAcc', 'categorize_TestAcc',
       'categorize_CumulativeUserItemAcc',
       'categorize_CumulativeUserTagAverageAcc',
       'categorize_CumulativeUserTagExponentialAverage']

Feature = Feature + Categorical_Feature


# as category: integer여도 범주형으로 취급 가능
for feature in Categorical_Feature:
       test[feature] = test[feature].astype('category')
       X[feature] = X[feature].astype('category')

feat = X.columns.tolist()

exclude_columns = [
    "Timestamp",
    "answerCode",
    "DayOfWeek",
    'WeekOfYear',
    'UserAvgSolvingTime',
    'PastItemCount',
    "user_tag_total_answer",
]
fitered_feat = [col for col in Feature if col in feat]
filtered_feat = [column for column in Feature if column not in exclude_columns]

X = X[filtered_feat+['answerCode']]

def objective_bestparam(trial, X):
    """
    best param을 fine-tuning하기!
    """
    df = X.copy()
    x_train, x_val, y_train, y_val = train_test_split(X.drop('answerCode', axis=1), X['answerCode'], test_size=0.2)

    # x_train = df.loc[train_idx].drop('answerCode')
    # y_train = df.loc[train_idx,'answerCode']
    train_set = lgb.Dataset(x_train, y_train)

    # x_val = df.loc[val_idx].drop('answerCode')
    # y_val = df.loc[val_idx,'answerCode']
    val_set = lgb.Dataset(x_val, y_val)

    params_LGBM = {
        'random_state': 42,
        'objective': 'binary',  # 이진 분류 문제
        'metric': 'auc',  # 평가 지표로 AUC 사용
        'boosting_type': 'gbdt',  # gbdt는 일반적인 그래디언트 부스팅 결정 
        'num_round' : trial.suggest_int('num_round', 1000, 5000),  
        
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),  # 트리의 최대 리프 노드 개수
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.4),  # 학습 속도
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),  # 각 트리에 사용할 특성의 비율
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),  # 리프 노드에 필요한 최소 데이터 수
        'max_depth': trial.suggest_int('max_depth', 3, 50),  # 트리의 최대 깊이
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),  # L1 정규화 강도
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 15.0),  # L2 정규화 강도
        'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0),  # 분할 최소 이득
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 1e+2),  # 자식 노드에서 필요한 최소 가중치 합계
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),  # 각 트리에 사용할 데이터의 비율
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),  # 데이터 샘플링 빈도
        'max_bin': trial.suggest_int('max_bin', 32, 512),  # 히스토그램 분할 중 최대 bin 수
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),  # 양성 클래스의 가중치
        'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 50.0),  # 카테고리 특징을 부드럽게 하는 파라미터
        }


    model = lgb.train(
            params_LGBM, # parameter 입력
            train_set,
            valid_sets=[train_set, val_set],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(30),
            ],
            categorical_feature = ['userID', 'assessmentItemID', 'testId','KnowledgeTag',
                        'Month', 'TimeOfDay', 
        'UserRecentTagAnswer',
        'UserRecentItemAnswer',
        'categorize_solvingTime',
        'categorize_ItemAcc', 'categorize_TagAcc', 'categorize_TestAcc',
        'categorize_CumulativeUserItemAcc',
        'categorize_CumulativeUserTagAverageAcc',
        'categorize_CumulativeUserTagExponentialAverage']
        )

    preds = model.predict(x_val)
    auc = roc_auc_score(y_val, preds)

    return auc



# def objective(trial, model, X, y, iterations=5):

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
#     f1 = trial.suggest_uniform('f1', 0.1, 1.0)
#     f2 = trial.suggest_uniform('f2', 0.1, 3)
#     f3 = trial.suggest_int('f3', 20, 100)
#     f4 = trial.suggest_int('f4', 20, 50)
#     f5 = trial.suggest_int('f5', 1, 5)
#     lr_factor = trial.suggest_uniform('lr_factor', 0.1, 0.7)
    
    
#     params = lgbm_params.copy()
#     print(f'RMSE for base model is {np.sqrt(mean_squared_error(y_val, Model.predict(X_val)))}')

#     for i in range(1, iterations):
#         if i > 2:
#             params['reg_lambda'] *=  f1
#             params['reg_alpha'] += f2
#             params['num_leaves'] += f3
#             params['min_child_samples'] -= f4
#             params['cat_smooth'] -= f5
#             params['learning_rate'] *= lr_factor
#             #params['max_depth'] += f5

       
#         params['learning_rate'] = params['learning_rate'] if params['learning_rate'] > 0.0009 else 0.0009
#         # need to stop learning rate to reduce to a very insignificant value, hence we use this threshold
        
#         Model = model(**params).fit(X_train, y_train, eval_set=[(X_val, y_val)],
#                           eval_metric=['rmse'],
#                           early_stopping_rounds=200, 
#                           categorical_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                           verbose=1000,
#                           init_model=Model if i > 1 else lgbm)# we will use pre trained model for first iteration
     
#         print(f'RMSE for {i}th model is {np.sqrt(mean_squared_error(y_val, Model.predict(X_val)))}')
           
              
#     RMSE = mean_squared_error(y_val, Model.predict(X_val), squared=False)
#     return RMSE


study = optuna.create_study(direction='maximize', study_name='lgbm', sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
study.optimize(lambda trial: objective_bestparam(trial=trial, X=X), n_trials=50)

print(study.best_params)

korea = pytz.timezone("Asia/Seoul")
current_time = datetime.now(korea).strftime("%m-%d %H:%M")

with open(f'log/lgbm/{current_time}.json', 'w') as f:
    json.dump(study.best_params, f)