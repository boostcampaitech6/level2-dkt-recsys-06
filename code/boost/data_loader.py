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
        df = self.train
        train = df[df["answerCode"] >= 0]
        test = df[df["answerCode"] == -1]
        data["train"], data["test"] = train, test
        return data
    
    
    def split_data(self, args) -> dict:
        """
        data의 구성
        data['train'] : 전체 user_id에 대한 데이터(Test에 있는 User에 대해서는 이미 마지막으로 푼 문제 정보가 없음)
        data['train_split'] : 전체 user_id별 마지막으로 푼 문제를 제외한 데이터
        data['valid'] : 전체 user_id별 마지막으로 푼 문제에 대한 데이터

        validation window: 몇 개의 last window로 검증한건지
        -> data[f'train_{i}']
        """
        data = self.restruct_data()
        FE_train= type_conversion(data['train'], self.args)

        train = data['train'].copy()
        # window 수 만큼 iteration
        for i in range(self.args.n_window):            
            print(f'{i}the window has {train.shape[0]} data')
            train["is_valid"] = [False] * train.shape[0]
            idx_last = train.drop_duplicates(subset="userID", keep="last").index
            train.loc[idx_last, "is_valid"] = True

            train, valid = train[train["is_valid"] == False], train[train["is_valid"] == True]
            data[f'train_{i}'] = train.drop("is_valid", axis=1)
            data[f'valid_{i}'] = valid.drop("is_valid", axis=1)

        return data, FE_train

def type_conversion(df, args):
        le = preprocessing.LabelEncoder()

        # [FEAT] integer여도 범주형으로 취급 가능
        for feature in args.cat_feats:
                df[feature] = df[feature].astype('category')

        for feature in df.columns:
            if df[feature].dtypes == "object" or df[feature].dtypes == "UInt32":
                df[feature] = le.fit_transform(df[feature])
        return df

class Preprocess:
    def __init__(self, args, data: dict):
        self.args = args
        self.data = data

    def preprocess(self) -> dict:

        for i in range(self.args.n_window):

            self.data[f"train_{i}_x"] = self.data[f"train_{i}"].drop("answerCode", axis=1)
            self.data[f"train_{i}_y"] = self.data[f"train_{i}"]["answerCode"]

            self.data[f"valid_{i}_x"] = self.data[f"valid_{i}"].drop("answerCode", axis=1)
            self.data[f"valid_{i}_y"] = self.data[f"valid_{i}"]["answerCode"]

        self.data["test"] = self.data["test"].drop("answerCode", axis=1)



        # as category: integer여도 범주형으로 취급 가능
        for i in range(self.args.n_window):
            for state in [f"train_{i}_x", f"valid_{i}_x", "test"]:
                df = self.data[state]
                
                for feature in self.args.cat_feats:
                    df[feature] = df[feature].astype('category')

        # LE: data['train']학습 -> split 별로 train_{i}, val_{i}, test transform
        for feature in df.columns:
            le = preprocessing.LabelEncoder()
            le.fit(self.data['train'][feature])

            for i in range(self.args.n_window):
                for state in [f"train_{i}_x", f"valid_{i}_x"]:
                    df = self.data[state]
                    if df[feature].dtypes == "object" or df[feature].dtypes == "UInt32":
                        print(f'encoded {feature}')
                        df[feature] = le.transform(df[feature])
                    self.data[state] = df
            if self.data['test'][feature].dtypes == "object" or self.data['test'][feature].dtypes == "UInt32":
                self.data['test'][feature] = le.transform(self.data['test'][feature])

        return self.data
    


