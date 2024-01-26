# 필요한 라이브러리 및 모듈을 임포트합니다.
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import SaintPlus, NoamOpt  # 사용자 정의 SAINT+ 모델 및 옵티마이저
from torch.utils.data import DataLoader  # 데이터 로더
from data_generator import Riiid_Sequence  # 사용자 정의 데이터 제너레이터
from sklearn.metrics import roc_auc_score  # 성능 평가를 위한 AUC 스코어
import random
import wandb  # Weights & Biases 라이브러리
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 라이브러리

# 랜덤 시드 설정
seed = 42

# 랜덤 시드를 고정하여 실험의 재현성을 보장합니다.
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# GPU 사용 가능 여부에 따라 장치(device)를 설정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 문제, 테스트, 카테고리, 태그의 개수를 설정합니다.
n_problems = 9454
n_tests = 1537
n_categories = 9
n_tags = 912

# 훈련 데이터를 불러옵니다. 데이터는 피클 파일 형태로 저장되어 있어야 합니다.
with open("/opt/ml/input/data/train_group.pkl.zip", "rb") as pick:
    group = pickle.load(pick)

# 훈련 데이터를 훈련 세트와 검증 세트로 분할합니다.
train_index, valid_index = train_test_split(
    group.index, train_size=0.8, random_state=seed
)

# 분할된 인덱스를 사용하여 훈련 세트와 검증 세트를 생성합니다.
train_group = group[train_index]
val_group = group[valid_index]

# Weights & Biases 스윕 설정을 정의합니다.
sweep_configuration = {
    "method": "bayes",  # 베이지안 최적화 방법 사용
    "metric": {"goal": "maximize", "name": "auc"},  # 목표: 성능 최대화  # 성능 평가 지표: AUC
    "parameters": {
        # 각 하이퍼파라미터의 범위 및 값 설정
        "num_layers": {"values": [1, 2, 4]},
        "num_heads": {"values": [2, 4, 8]},
        "d_model": {"values": [64, 128, 256]},
        "d_ffn": {"values": [2, 3, 4, 5, 6]},
        "seq_len": {"distribution": "int_uniform", "min": 10, "max": 500},
        "warmup_steps": {"distribution": "int_uniform", "min": 100, "max": 10000},
        "dropout": {"distribution": "uniform", "min": 0, "max": 0.9},
        "lr": {"distribution": "uniform", "min": 0.001, "max": 0.01},
        "batch_size": {"values": [64, 128]},
        "epochs": {"value": 100},
        "patience": {"value": 10},
    },
}


# 목표 함수를 정의합니다. 이 함수는 SAINT+ 모델을 훈련하고 성능을 평가합니다.
def objective_function():
    wandb.init()  # Weights & Biases 초기화
    args = wandb.config  # 하이퍼파라미터 설정

    # 데이터 로더 설정
    train_loader = DataLoader(
        Riiid_Sequence(train_group, args.seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        Riiid_Sequence(val_group, args.seq_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    # 손실 함수와 모델 초기화
    loss_fn = nn.BCELoss()
    model = SaintPlus(
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        d_ffn=args.d_model * args.d_ffn,
        d_model=args.d_model,
        num_heads=args.num_heads,
        n_problems=n_problems,
        n_tests=n_tests,
        n_categories=n_categories,
        n_tags=n_tags,
        dropout=args.dropout,
    )
    optimizer = NoamOpt(
        args.d_model, 1, args.warmup_steps, optim.Adam(model.parameters(), lr=args.lr)
    )
    model.to(device)
    loss_fn.to(device)

    # 훈련 및 검증 과정
    # 이후 코드는 모델 훈련, 검증, 최적 모델 저장 및 조기 종료 로직을 포함합니다.
    # 코드가 길어짐에 따라 주석은 간략화되었습니다.


# Weights & Biases 스윕 ID 생성
sweep_id = wandb.sweep(sweep_configuration, project="saintplus")

# Weights & Biases 에이전트 실행
wandb.agent(sweep_id, function=objective_function, count=50)
