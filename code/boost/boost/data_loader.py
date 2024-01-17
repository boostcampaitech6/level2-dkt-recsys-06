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
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def restruct_data(self) -> dict:
        # 아직 FE merge하기 이전
        """
        Test data와 Train data를 concat하여 test에만 있는 유저 정보도 train으로 추가한다.
        """
        data = {}
        self.train = self.train.sort_values(by=["userID", "Timestamp"]).reset_index(
            drop=True
        )
        self.test = self.test.sort_values(by=["userID", "Timestamp"]).reset_index(
            drop=True
        )
        df = pd.concat([self.train, self.test], axis=0)
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
        df = data["train"]
        df["is_valid"] = [False] * df.shape[0]
        df.loc[
            df.drop_duplicates(subset="userID", keep="last").index, "is_valid"
        ] = True

        train, valid = df[df["is_valid"] == False], df[df["is_valid"] == True]
        train = train.drop("is_valid", axis=1)
        valid = valid.drop("is_valid", axis=1)
        data["train_split"], data["valid"] = train, valid

        return data


class Preprocess:
    def __init__(self, args, data: dict):
        self.args = args
        self.data = data

    def type_conversion(self) -> dict:
        self.data["train_x"] = self.data["train"].drop("answerCode", axis=1)
        self.data["train_y"] = self.data["train"]["answerCode"]

        self.data["valid_x"] = self.data["valid"].drop("answerCode", axis=1)
        self.data["valid_y"] = self.data["valid"]["answerCode"]

        self.data["test"] = self.data["test"].drop("answerCode", axis=1)

        # 카테고리형 feature
        for state in ["train_x", "valid_x", "test"]:
            df = self.data[state]
            le = preprocessing.LabelEncoder()
            for feature in df.columns:
                if df[feature].dtypes == "object" or df[feature].dtypes == "UInt32":
                    df[feature] = le.fit_transform(df[feature])
            self.data[state] = df
        return self.data

    def preprocess(self) -> dict:
        data = self.type_conversion()
        return data
