import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing


class Dataset:
    def __init__(self, train: pd.DataFrame, args):
        self.train = train
        self.args = args

    def restruct_data(self) -> dict:
        # train과 test 분할
        data = {}
        df = self.train.sort_values(by=["userID", "Timestamp", "assessmentItemID"]).reset_index(drop=True)
        train = df[df["answerCode"] >= 0]
        test = df[df["answerCode"] == -1]
        data["train"], data["test"] = train, test
        return data
    
    
    def split_data(self) -> dict:
        """
        data의 구성
        data['train'] : 전체 user_id에 대한 데이터(Test에 있는 User에 대해서는 이미 마지막으로 푼 문제 정보가 없음)
        data['train_split'] : 전체 user_id별 마지막으로 푼 문제를 제외한 데이터
        data['valid'] : 전체 user_id별 마지막으로 푼 문제에 대한 데이터
        """
        data = self.restruct_data()
        
        train = data['train'].copy()
        train["is_valid"] = [False] * train.shape[0]
        idx_last = train.drop_duplicates(subset="userID", keep="last").index
        train.loc[idx_last, "is_valid"] = True

        train, valid = train[train["is_valid"] == False], train[train["is_valid"] == True]
        data['train'] = train.drop("is_valid", axis=1)
        data['valid'] = valid.drop("is_valid", axis=1)

        print(f'{data[f"train"].shape[0]} train data')
        print(f'{data[f"valid"].shape[0]} valid data')

        data["train_x"] = data["train"][self.args.feature]
        data["train_y"] = data["train"]["answerCode"]

        data["valid_x"] = data["valid"][self.args.feature]
        data["valid_y"] = data["valid"]["answerCode"]

        data["test"] = data["test"][self.args.feature]

        # as category: integer여도 범주형으로 취급 가능
        for state in [f"train_x", f"valid_x", "test"]:
            df = data[state]
                
            for feature in self.args.cat_feats:
                df[feature] = df[feature].astype('category')


        return data

