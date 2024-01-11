import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(
        self, data: np.ndarray, ratio: float = 0.7, shuffle: bool = True, seed: int = 0
    ) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    # cat: label encoding, unknown처리
    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        # feature engineering 위해 arg.cat_feats로 feature 목록선택
        # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        # cate_cols = list(set(self.args.base_cat_feats + self.args.cat_feats))
        cate_cols = self.args.cat_feats

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        # testId, assessmentItemID, KnowledgeTag에 대해서 unknown 토큰 처리
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # timestamp -> timetuple
        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed

        return df

    # userID 안들어감! -> 문항, 시험지, 문항 풀기시작한 정보만 들어감
    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)  # feature 선택
        df = self.__preprocessing(df, is_train)  # 범주형 전처리

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        # 최종 피처선택
        # columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        columns = self.args.feats
        print(f'-----------------------\ncolumns:{columns}\n------------------------')
        group = (
            df[columns]
            .groupby("userID")
            .apply(lambda r: tuple([r[col].values for col in columns[1:]]))
        )
        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len
        self.args = args

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # 반드시 args의 길이 앞뒤 맞아야 해! feats = 3+new_cat_feats+new_num_feats
        # Load from data: 순서는 0:test, 1:quest, 2:tag, 3:correct는 고정! 나머지는 new로!
        question, test, correct, tag = row[0], row[1], row[2], row[3]
        new_cat_feats = row[4 : 4 + len(self.args.new_cat_feats)]  # 4번째부터 새로운 범주형
        new_num_feats = row[
            4 + len(self.args.new_cat_feats) :
        ]  # 뒤쪽 수치형 (-0때문에 -n slicing X)

        # loader 넘어가기 위해 준비
        data = {
            "test": torch.LongTensor(test + 1),
            "question": torch.LongTensor(question + 1),
            "tag": torch.LongTensor(tag + 1),
            "correct": torch.LongTensor(correct),
        }

        # 새로운 피처들이 있다면 data에 추가해라
        if len(new_cat_feats) > 0:
            for i, cat_feat in enumerate(new_cat_feats):
                data[f"new_cat_feats_{i}"] = torch.LongTensor(cat_feat)
        if len(new_num_feats) > 0:
            for i, num_feat in enumerate(new_num_feats):
                data[f"new_num_feats_{i}"] = torch.FloatTensor(num_feat)

        # Generate mask: max seq len보다 길면 자르고 짧으면 그냥 둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len, dtype=torch.int64)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = torch.LongTensor(tmp)
            mask = torch.zeros(self.max_seq_len, dtype=torch.int64)
            mask[-seq_len:] = 1
        data["mask"] = mask

        # Generate interaction: 이전 문제 정답여부 (0:padding, 1:오답, 2:정답)
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)  # 이전 문제의 정답 여부
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = torch.LongTensor(interaction)

        data = {k: v.int() for k, v in data.items()}

        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(
    args, train: np.ndarray, valid: np.ndarray
) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
