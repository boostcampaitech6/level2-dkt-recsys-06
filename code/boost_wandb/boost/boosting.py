import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import wandb
from .utils import get_logger, logging_conf

import catboost as cat
import xgboost as xgb
import lightgbm as lgbm

logger = get_logger(logger_conf=logging_conf)


class boosting_model:
    def __init__(self, args, FEATURE, data):
        self.args = args
        self.feature = FEATURE
        self.feature.remove("answerCode") # remove for X
        self.data = data

        if args.model == "CAT":
            self.model = cat.CatBoostRegressor(
                learning_rate=self.args.learning_rate,
                iterations=self.args.n_estimators,
                random_state=self.args.random_state,
                max_depth=self.args.max_depth,
                reg_lambda=self.args.reg_lambda,
                early_stopping_rounds=self.args.early_stopping_rounds,
                min_data_in_leaf=self.args.min_data_in_leaf,
                task_type="GPU",
                devices="cuda",
                eval_metric="AUC",
                has_time=True,
            )
        elif args.model == "XG":
            self.model = xgb.XGBRegressor(
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.n_estimators,
                random_state=self.args.random_state,
                max_depth=self.args.max_depth,
                reg_lambda=self.args.reg_lambda,
                subsample=self.args.subsample,
                early_stopping_rounds=self.args.early_stopping_rounds,
                max_leaves=self.args.max_leaves,
            )
        elif args.model == "LGBM":
            self.model = lgbm.LGBMRegressor(
                learning_rate=self.args.learning_rate,
                num_round=self.args.n_estimators,
                random_state=self.args.random_state,
                max_depth=self.args.max_depth,
                reg_lambda=self.args.reg_lambda,
                subsample=self.args.subsample,
                early_stopping_rounds=self.args.early_stopping_rounds,
                num_leaves=self.args.max_leaves,
                min_data_in_leaf=self.args.min_data_in_leaf,
                feature_fraction=self.args.feature_fraction,
                bagging_freq=self.args.bagging_freq,
                categorical_feature=args.cat_feats,
                objective='binary',
                metric='auc'
            )
        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")

    def training(self, data, args, FEATURE):
        logger.info("###start MODEL training ###")
        logger.info(args.model)
        logger.info(self.feature)

        if args.model == "CAT":
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                cat_features=self.feature,
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                verbose=100,
            )
        elif args.model == "XG":
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                verbose=100,
                eval_metric="auc",
            )
        else: #LGBM
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                eval_metric="AUC",
            )
        y_pred = self.model.predict(data["valid_x"][self.feature])
        auc = accuracy_score(data["valid_y"], np.where(y_pred >= 0.5, 1, 0))
        acc = roc_auc_score(data["valid_y"], np.where(y_pred >= 0.5, 1, 0))

        wandb.log(
            dict(
                model=args.model,
                train_acc=acc,
                train_auc=auc
            )
        )
        logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)

    def inference(self, data, save_time):
        # submission 제출하기 위한 코드
        print("### Inference && Save###")
        test_pred = self.model.predict(data["test"][self.feature])
        data["test"]["prediction"] = test_pred
        submission = data["test"]["prediction"].reset_index(drop=True).reset_index()
        submission.rename(columns={"index": "id"}, inplace=True)
        submission_filename = f"{self.args.model}_{save_time}_submission.csv"
        submission.to_csv(
            os.path.join(self.args.output_dir, submission_filename), index=False
        )
