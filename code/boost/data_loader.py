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
    
    
    def split_data(self) -> dict:
        """
        data의 구성
        data['train'] : 전체 user_id에 대한 데이터(Test에 있는 User에 대해서는 이미 마지막으로 푼 문제 정보가 없음)
        data['train_split'] : 전체 user_id별 마지막으로 푼 문제를 제외한 데이터
        data['valid'] : 전체 user_id별 마지막으로 푼 문제에 대한 데이터
        """
        data = self.restruct_data()
        FE_train= type_conversion(data["train"], self.args) # FE_train이 뒤 코드에서 안쓰임

        df = data["train"]
        df["is_valid"] = [False] * df.shape[0]
        df.loc[
            df.drop_duplicates(subset="userID", keep="last").index, "is_valid"
        ] = True

        train, valid = df[df["is_valid"] == False], df[df["is_valid"] == True]
        train = train.drop("is_valid", axis=1)
        valid = valid.drop("is_valid", axis=1)
        data["train_split"], data["valid"] = train, valid  
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
        self.data["train_x"] = self.data["train"].drop("answerCode", axis=1)
        self.data["train_y"] = self.data["train"]["answerCode"]

        self.data["valid_x"] = self.data["valid"].drop("answerCode", axis=1)
        self.data["valid_y"] = self.data["valid"]["answerCode"]

        self.data["test"] = self.data["test"].drop("answerCode", axis=1)

        # 카테고리형 feature -> 정수
        for state in ["train_x", "valid_x", "test"]:
            df = self.data[state]
            le = preprocessing.LabelEncoder()

            # [FEAT] integer여도 범주형으로 취급 가능
            for feature in self.args.cat_feats:
                df[feature] = df[feature].astype('category')

            for feature in df.columns:
                if df[feature].dtypes == "object" or df[feature].dtypes == "UInt32":
                    df[feature] = le.fit_transform(df[feature])
            self.data[state] = df
        return self.data
    


