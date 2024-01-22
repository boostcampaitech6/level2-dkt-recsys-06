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
    def __init__(self, train: pd.DataFrame,args):
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
    
    
    def split_data(self,args) -> dict:
        """
        data의 구성
        data['train'] : 전체 user_id에 대한 데이터(Test에 있는 User에 대해서는 이미 마지막으로 푼 문제 정보가 없음)
        data['train_split'] : 전체 user_id별 마지막으로 푼 문제를 제외한 데이터
        data['valid'] : 전체 user_id별 마지막으로 푼 문제에 대한 데이터
        """
        data = self.restruct_data()
        FE_train= type_conversion(data["train"],args)

        train = data['train'].copy()
        test_users = data['test']['userID'].unique()
        
       # window 수 만큼 iteration
        for i in range(self.args.n_window):
            # 'train' DataFrame을 'userID' 및 'testId'로 그룹화하고 각 그룹 내에서 'Timestamp'의 최댓값의 인덱스를 찾습니다
            last_solved_indices = train.groupby(['userID', 'testId'])['Timestamp'].idxmax()

            # 위에서 찾은 인덱스를 기반으로 부울 마스크를 생성합니다
            is_valid_mask = train.index.isin(last_solved_indices)

            # 'train_i' 및 'valid_i'라는 새로운 DataFrame을 만들어 'train' DataFrame의 하위 집합을 복사합니다
            data[f'train_{i}'] = train[~is_valid_mask].copy()
            data[f'valid_{i}'] = train[is_valid_mask].copy()

            # 'train_i' 및 'valid_i' DataFrame에서 임시 'is_valid' 열을 삭제합니다
            data[f'train_{i}'].drop(columns='is_valid', inplace=True, errors='ignore')
            data[f'valid_{i}'].drop(columns='is_valid', inplace=True, errors='ignore')

            print(f'{i} window has {data[f"train_{i}"].shape[0]} train data')
            print(f'{i} window has {data[f"valid_{i}"].shape[0]} valid data')

        return data, FE_train

def type_conversion(df,args):
        le = preprocessing.LabelEncoder()

        # [FEAT] integer여도 범주형으로 취급 가능
        for feature in args.cat_feats:
                df[feature] = df[feature].astype('category')

        #for feature in df.columns:
            #if df[feature].dtypes == "object" or df[feature].dtypes == "UInt32":
                #df[feature] = le.fit_transform(df[feature])
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

        for feature in df.columns:
            le = preprocessing.LabelEncoder()
            le.fit(self.data['train'][feature])

            for i in range(self.args.n_window):
                for state in [f"train_{i}_x", f"valid_{i}_x"]:
                    df = self.data[state]
                    if df[feature].dtypes == "object":
                        print(f'encoded {feature}')
                        df[feature] = le.transform(df[feature])
                    self.data[state] = df
            if self.data['test'][feature].dtypes == "object":
                self.data['test'][feature] = le.transform(self.data['test'][feature])


        return self.data
    


