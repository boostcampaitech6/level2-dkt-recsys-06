import os
from typing import Tuple

import pandas as pd
import torch

from lightgcn.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def prepare_dataset(device: str, data_dir: str) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir)
    train_data, test_data = separate_data(data=data) # train_data: 이전 기록들 / test_data:마지막 기록 (answerCode:-1)
    id2index: dict = indexing_data(data=data)
    train_data_proc = process_data(data=train_data, id2index=id2index, device=device)
    test_data_proc = process_data(data=test_data, id2index=id2index, device=device)

    print_data_stat(train_data, "Train")
    print_data_stat(test_data, "Test")

    return train_data_proc, test_data_proc, len(id2index), id2index

<<<<<<< HEAD

=======
# GCN은 train, test 이미 합쳐져있네
>>>>>>> wooksbaby
def load_data(data_dir: str) -> pd.DataFrame:
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

<<<<<<< HEAD
    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
=======
    data = pd.concat([data1, data2]) # train과 test 합치기
    data['userID'] = data['userID'].astype(str)
    data['assessmentItemID'] = data['assessmentItemID'].astype(str)
    # data['answerCode'] = data['answerCode'].astype(int)
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    ) # 최신 데이터 남기기 (동일한 문제에 대해)
>>>>>>> wooksbaby
    return data

# X(train+test),Y(test 마지막)
def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0] # test의 마지막 아닌 문제는 이쪽으로 들어가지 않나요?
    test_data = data[data.answerCode < 0] # -1: test의 각 유저의 마지막 문제? (task)
    return train_data, test_data

# user_id 그 뒤에 item_id -> index
def indexing_data(data: pd.DataFrame) -> dict:
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid2index = {v: i for i, v in enumerate(userid)} # user index: 0~len(user)-1
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)} # user index: len(user)~
    id2index = dict(userid2index, **itemid2index) #user와 item id 사전
    return id2index


# user, assessmentID만을 쓰네
def process_data(data: pd.DataFrame, id2index: dict, device: str) -> dict:
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id2index[user], id2index[item]
        edge.append([uid, iid]) # 유저와 문제 id만 들어가!
        label.append(int(acode))

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data: pd.DataFrame, name: str) -> None:
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
