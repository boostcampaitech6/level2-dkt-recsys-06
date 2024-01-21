import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold

import pickle


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

    def kfold(self, data: np.ndarray, k: int = 5, shuffle: bool = True):
        # s_kfold = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=self.args.seed)
        # folds = s_kfold.split(X=data[:,1:], y=data[:,:1])
        kfold = KFold(n_splits=k, shuffle=True, random_state=self.args.seed)
        folds = kfold.split(data)
        return folds

    def manual_kfold(self, data: np.ndarray, k: int = 5, shuffle: bool = True):
        # 유저별로 각 fold에 나누기
        folds = {}
        for i in range(1, k + 1):
            folds[f"fold_{i}"] = []
        for i in range(len(data)):
            folds[f"fold_{i%k+1}"].append(data[i])

        # cross validation
        fold_iters = []
        for i in range(1, k + 1):
            train = []
            val = []
            for j in range(1, k + 1):
                if i == j:
                    val.extend(folds[f"fold_{j}"])
                else:
                    train.extend(folds[f"fold_{j}"])
            fold_iters.append([train, val])

        return fold_iters

    def __slidding_window(self, data, args):
        """
        data shape: [n_users, n_feats, original_seq_len]
        """

        window_size = args.max_seq_len
        stride = args.stride

        augmented_datas = []
        for row in data:  # user마다
            seq_len = len(row[0])  # row:[n_feats, seq_len]

            user_augmented = []
            # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
            if seq_len <= window_size:
                user_augmented.append(row)
            else:
                total_window = ((seq_len - window_size) // stride) + 1  # 윈도우 갯수

                # slidding window 적용
                for window_i in range(total_window):
                    # window로 잘린 데이터를 모으는 리스트
                    window_data = []
                    for col in row:  # col:[seq_len]
                        # window_data.append(col[window_i*stride:window_i*stride + window_size]) # 앞에서부터
                        if window_i == 0:
                            window_data.append(
                                col[-1 * window_i * stride - window_size :]
                            )  # 뒤에서부터
                        else:
                            window_data.append(
                                col[
                                    -1 * window_i * stride
                                    - window_size : -1 * window_i * stride
                                ]
                            )  # 뒤에서부터

                    # Shuffle
                    # 마지막 데이터의 경우 shuffle을 하지 않는다
                    if args.shuffle and window_i + 1 != total_window:
                        shuffle_datas = self.__shuffle(window_data, window_size, args)
                        user_augmented += shuffle_datas
                    else:
                        user_augmented.append(window_data)

                # slidding window에서 뒷부분이 누락될 경우 추가
                total_len = window_size + (stride * (total_window - 1))
                if seq_len != total_len:
                    window_data = []
                    for col in row:
                        window_data.append(col[-window_size:])
                    user_augmented.append(window_data)

                # slidding window에서 앞부분이 누락될 경우 추가
                # total_len = window_size + (stride * (total_window - 1))
                # if seq_len != total_len:
                #     window_data = []
                #     for col in row:
                #         window_data.append(col[:window_size])
                #     user_augmented.append(tuple(window_data)) #tuple을 list로 바꿈 오류 예상ㅇ # [window_num, n_feats, window_size]

                # split할 때 유저별로 fold될 수 있도록 labeling이나 순서 정해주자
                # => 이거 기준으로 k-fold : 이렇게 되면, 기존=유저당 1개 seq 마지막 loss 학습 / 이후=유저당 (k-1)*fold개 sequence 마지막 loss 학습

                # user 별로 random choice
                if self.args.n_choice != 0:
                    # n_choice 보다 많으면 -> random choice
                    if self.args.n_choice <= len(user_augmented):
                        idx = np.random.choice(
                            np.arange(len(user_augmented)),
                            size=self.args.n_choice,
                            replace=False,
                        )
                        augmented_datas += [user_augmented[i] for i in idx]
                    # n_choice 보다 적으면 -> 최신 window부터 shuffle로 데이터 추가
                    else:
                        shuffle_size = self.args.n_choice - len(user_augmented)
                        for i in range(shuffle_size):
                            temp = user_augmented[i]  # temp: [n_feats, window_size]
                            for col in temp:  # col:[seq_len]
                                random.shuffle(col)
                            user_augmented.append(temp)
                        augmented_datas += user_augmented

                else:
                    augmented_datas += list(user_augmented)

        return augmented_datas  # [n_user*n_choice, n_feats, window_size]

    def __shuffle(self, data, data_size, args):
        shuffle_datas = []
        for i in range(args.shuffle_n):
            # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가 #1개의 window -> n개의 window로 둔갑!
            shuffle_data = []
            random_index = np.random.permutation(data_size)
            for col in data:
                shuffle_data.append(col[random_index])
            shuffle_datas.append(tuple(shuffle_data))
        return shuffle_datas

    def __data_augmentation(self, data, args):
        if args.window == True:
            data = self.__slidding_window(data, args)
        return data

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    # cat: label encoding, unknown처리
    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = [
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
        ] + self.args.new_cat_feats

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        df["quiz"] = df["assessmentItemID"]

        # testId, assessmentItemID, KnowledgeTag에 대해서 unknown 토큰 처리
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]  # col: 가짓수?
                le.fit(a)  # 정렬 순으로 labeling해!
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
    def load_data_from_file(
        self, args, file_name: str, is_train: bool = True
    ) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)

        df = df[df.answerCode >= 0]
        # 새로운 범주형 피처에 대해 자동으로 input의 가짓 수 (one-hot의 차원 수) 계산
        for cat in args.new_cat_feats:
            if args.n_cat_feats:
                args.n_cat_feats.append(df[cat].nunique())
            else:
                args.n_cat_feats = [df[cat].nunique()]

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

        if self.args.graph_embed:
            print(
                f"-----------------------\ncolumns:{columns + ['graph']}\n------------"
            )
        else:
            print(
                f"-----------------------\ncolumns:{columns}\n------------------------"
            )
        group = (
            df[columns + ["quiz"]]
            .groupby("userID")
            .apply(lambda r: tuple([r[col].values for col in columns[1:] + ["quiz"]]))
        ).values

        if self.args.window:
            group = self.__data_augmentation(
                group, self.args
            )  # [n_users, n_feats, seq_len] -> [n_users, n_feats, seq_len]

        return group  # shape: [n_users*n_choice, n_feats, origianl_seq]

    def load_train_data(self, args, file_name: str) -> None:
        self.train_data = self.load_data_from_file(args, file_name)

    def load_test_data(self, args, file_name: str) -> None:
        self.test_data = self.load_data_from_file(args, file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args, dict_graph):
        self.data = data
        self.max_seq_len = args.max_seq_len
        self.args = args
        self.dict_graph = dict_graph

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # 반드시 args의 길이 앞뒤 맞아야 해! feats = 3+new_cat_feats+new_num_feats
        # Load from data: 순서는 0:question, 1:test, 2:correct 3:tag는 고정! 나머지는 new로!
        correct, question, test, tag = row[0], row[1], row[2], row[3]
        new_cat_feats = row[4 : 4 + len(self.args.new_cat_feats)]  # 4번째부터 새로운 범주형
        num_feats = row[
            4
            + len(self.args.new_cat_feats) : 4
            + len(self.args.new_cat_feats)
            + len(self.args.new_num_feats)
        ]  # 뒤쪽 수치형 (-0때문에 -n slicing X)

        if self.args.graph_embed:
            quiz = row[-1]
            embed_graph = [self.dict_graph[id] for id in quiz]

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
        if len(num_feats) > 0:
            for i, num_feat in enumerate(num_feats):
                data[f"num_feats_{i}"] = torch.FloatTensor(num_feat)
        if self.args.graph_embed:
            data["embed_graph"] = torch.FloatTensor(embed_graph)

        # Generate mask: max seq len보다 길면 자르고 짧으면 그냥 둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for i, (k, seq) in enumerate(data.items()):
                # Pre-padding non-valid sequences

                if i < (len(self.args.cat_feats) - 1):  # 범주형 (-1: userID)
                    tmp = torch.zeros(self.max_seq_len, dtype=torch.int64)

                    tmp[self.max_seq_len - seq_len :] = data[k]
                    data[k] = torch.LongTensor(tmp)
                elif k == "embed_graph":
                    tmp = torch.zeros((self.max_seq_len, 64), dtype=torch.float32)
                    tmp[self.max_seq_len - seq_len :] = data[k]
                    data[k] = tmp
                else:  # 수치형
                    tmp = torch.zeros(self.max_seq_len, dtype=torch.float32)
                    tmp[self.max_seq_len - seq_len :] = data[k]
                    data[k] = tmp
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

        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(
    args, train: np.ndarray, valid: np.ndarray, dict_graph: dict
) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args, dict_graph)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args, dict_graph)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
