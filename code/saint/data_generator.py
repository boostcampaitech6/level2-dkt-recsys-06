# 필요한 라이브러리를 임포트합니다.
import gc
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Riiid_Sequence 클래스는 PyTorch의 Dataset 클래스를 상속받아 구현됩니다.
# 이 클래스는 학습 데이터를 위한 커스텀 데이터셋을 생성합니다.
class Riiid_Sequence(Dataset):
    def __init__(self, groups, seq_len):
        self.samples = {}  # 사용자별 샘플을 저장할 딕셔너리
        self.seq_len = seq_len  # 시퀀스 길이
        self.user_ids = []  # 사용자 ID 리스트

        # groups에 있는 각 사용자 데이터를 순회하면서 데이터셋을 생성합니다.
        for user_id in groups.index:
            # 사용자별 데이터를 추출합니다.
            (
                category,
                category_test,
                category_test_problem,
                problem_tag,
                problem_time,
                break_time,
                answer,
            ) = groups[user_id]

            # 최소 길이 조건을 만족하지 못하는 경우는 제외합니다.
            if len(category) < 2:
                continue

            # 시퀀스 길이보다 긴 데이터를 처리하는 로직입니다.
            if len(category) > self.seq_len:
                initial = len(category) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (
                        category[:initial],
                        category_test[:initial],
                        category_test_problem[:initial],
                        problem_tag[:initial],
                        problem_time[:initial],
                        break_time[:initial],
                        answer[:initial],
                    )
                chunks = len(category) // self.seq_len
                for c in range(chunks):
                    start = initial + c * self.seq_len
                    end = initial + (c + 1) * self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.samples[f"{user_id}_{c+1}"] = (
                        category[start:end],
                        category_test[start:end],
                        category_test_problem[start:end],
                        problem_tag[start:end],
                        problem_time[start:end],
                        break_time[start:end],
                        answer[start:end],
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    category,
                    category_test,
                    category_test_problem,
                    problem_tag,
                    problem_time,
                    break_time,
                    answer,
                )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        (
            category,
            category_test,
            category_test_problem,
            problem_tag,
            problem_time,
            break_time,
            answer,
        ) = self.samples[user_id]
        seq_len = len(category)

        # 시퀀스 길이에 맞게 데이터를 0으로 채워넣습니다.
        category_sample = np.zeros(self.seq_len, dtype=int)
        category_test_sample = np.zeros(self.seq_len, dtype=int)
        category_test_problem_sample = np.zeros(self.seq_len, dtype=int)
        problem_tag_sample = np.zeros(self.seq_len, dtype=int)
        problem_time_sample = np.zeros(self.seq_len, dtype=float)
        break_time_sample = np.zeros(self.seq_len, dtype=float)

        # 정답 및 라벨 데이터를 처리합니다.
        answer_sample = np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        # 데이터를 채워 넣습니다. 길이가 seq_len보다 짧은 경우 뒷부분에 채워넣습니다.
        if seq_len == self.seq_len:
            category_sample[:] = category
            category_test_sample[:] = category_test
            category_test_problem_sample[:] = category_test_problem
            problem_tag_sample[:] = problem_tag
            problem_time_sample[:] = problem_time
            break_time_sample[:] = break_time
            answer_sample[:] = answer
        else:
            category_sample[-seq_len:] = category
            category_test_sample[-seq_len:] = category_test
            category_test_problem_sample[-seq_len:] = category_test_problem
            problem_tag_sample[-seq_len:] = problem_tag
            problem_time_sample[-seq_len:] = problem_time
            break_time_sample[-seq_len:] = break_time
            answer_sample[-seq_len:] = answer

        # 마지막 시퀀스 요소를 라벨로 사용하고, 나머지는 입력 데이터로 사용합니다.
        category_sample = category_sample[1:]
        category_test_sample = category_test_sample[1:]
        category_test_problem_sample = category_test_problem_sample[1:]
        problem_tag_sample = problem_tag_sample[1:]
        problem_time_sample = problem_time_sample[1:]
        break_time_sample = break_time_sample[1:]
        label = answer_sample[1:]
        answer_sample = answer_sample[:-1]

        return (
            category_sample,
            category_test_sample,
            category_test_problem_sample,
            problem_tag_sample,
            problem_time_sample,
            break_time_sample,
            answer_sample,
            label,
        )


# Riiid_Sequence_Test 클래스는 Riiid_Sequence와 유사하지만 테스트 데이터를 위한 클래스입니다.
class Riiid_Sequence_Test(Dataset):
    def __init__(self, groups, seq_len):
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index:
            (
                category,
                category_test,
                category_test_problem,
                problem_tag,
                problem_time,
                break_time,
                answer,
            ) = groups[user_id]
            if len(category) < 2:
                continue

            if len(category) > self.seq_len:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    category[-seq_len:],
                    category_test[-seq_len:],
                    category_test_problem[-seq_len:],
                    problem_tag[-seq_len:],
                    problem_time[-seq_len:],
                    break_time[-seq_len:],
                    answer[-seq_len:],
                )
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    category,
                    category_test,
                    category_test_problem,
                    problem_tag,
                    problem_time,
                    break_time,
                    answer,
                )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        (
            category,
            category_test,
            category_test_problem,
            problem_tag,
            problem_time,
            break_time,
            answer,
        ) = self.samples[user_id]
        seq_len = len(category)

        category_sample = np.zeros(self.seq_len, dtype=int)
        category_test_sample = np.zeros(self.seq_len, dtype=int)
        category_test_problem_sample = np.zeros(self.seq_len, dtype=int)
        problem_tag_sample = np.zeros(self.seq_len, dtype=int)
        problem_time_sample = np.zeros(self.seq_len, dtype=float)
        break_time_sample = np.zeros(self.seq_len, dtype=float)

        answer_sample = np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            category_sample[:] = category
            category_test_sample[:] = category_test
            category_test_problem_sample[:] = category_test_problem
            problem_tag_sample[:] = problem_tag
            problem_time_sample[:] = problem_time
            break_time_sample[:] = break_time
            answer_sample[:] = answer
        else:
            category_sample[-seq_len:] = category
            category_test_sample[-seq_len:] = category_test
            category_test_problem_sample[-seq_len:] = category_test_problem
            problem_tag_sample[-seq_len:] = problem_tag
            problem_time_sample[-seq_len:] = problem_time
            break_time_sample[-seq_len:] = break_time
            answer_sample[-seq_len:] = answer

        category_sample = category_sample[1:]
        category_test_sample = category_test_sample[1:]
        category_test_problem_sample = category_test_problem_sample[1:]
        problem_tag_sample = problem_tag_sample[1:]
        problem_time_sample = problem_time_sample[1:]
        break_time_sample = break_time_sample[1:]
        label = answer_sample[1:]
        answer_sample = answer_sample[:-1]

        return (
            category_sample,
            category_test_sample,
            category_test_problem_sample,
            problem_tag_sample,
            problem_time_sample,
            break_time_sample,
            answer_sample,
            label,
        )
