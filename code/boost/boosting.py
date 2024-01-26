import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm
import optuna
import joblib
import json

# LGBM
def objective(trial,FEATURE,data):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42
    }

    # 데이터셋
    train_data = lgbm.Dataset(data["train_x"][FEATURE], label=data["train_y"])
    valid_data = lgbm.Dataset(data["valid_x"][FEATURE], label=data["valid_y"], reference=train_data)


    # LightGBM 모델 훈련
    num_round = 100
    bst = lgbm.train(params,train_data, num_round, valid_sets=valid_data)

    # 예측
    y_pred_proba = bst.predict(data["valid_x"][FEATURE], num_iteration=bst.best_iteration)
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_proba]

    # 정확도 및 AUC 계산
    accuracy = accuracy_score(data["valid_y"], y_pred_binary)
    auc = roc_auc_score(data["valid_y"], y_pred_proba)
    print('Accuracy: {:.4f}'.format(accuracy))
    print('AUC: {:.4f}'.format(auc))

    return 1 - accuracy  # 목적 함수는 최소화해야 하므로 (1 - accuracy)를 반환


class boosting_model:
    def __init__(self, args,FEATURE,data):
        self.args = args
        self.feature = FEATURE
        self.data = data
        
        if args.model == "CAT":
            self.model = CatBoostClassifier(learning_rate=self.args.learning_rate,
                n_estimators=self.args.n_estimators, task_type='GPU', devices='cuda' , eval_metric='AUC')
            
        elif args.model == "XG":
            self.model = xgb.XGBClassifier(
                learning_rate=self.args.learning_rate,
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
            )
        elif args.model == "LGBM":
            
            # Optuna 최적화
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial,self.feature, self.data), n_trials=50)

            # 최적 하이퍼파라미터 출력
            print('Hyperparameters: {}'.format(study.best_params))

            self.model = lgbm.LGBMClassifier(
               **study.best_params
            )

        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")

    def training(self, data, args, FEATURE):
        print("###start MODEL training ###")
        print(self.feature)
        if args.model == "CAT":
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=50,
                cat_features=list(data["train_x"][self.feature]),
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                verbose=10,
            )
            print(self.model.get_best_score())
            print(self.model.get_all_params())

            def display_feature_importance(importance, names, model_type):
                feature_importance = np.array(importance)
                feature_names = np.array(names)
    
                # Create a DataFrame to sort and display the results
                fi_df = pd.DataFrame({'Feature Names': feature_names, 'Feature Importance': feature_importance})
                fi_df = fi_df.sort_values(by='Feature Importance', ascending=False)

                # Print the results
                print(f"{model_type} Feature Importance:")
                print(fi_df)

            # 결과를 출력만 하는 함수 호출
            display_feature_importance(self.model.get_feature_importance(), FEATURE, 'CATBOOST')

        elif args.model == "XG":
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                early_stopping_rounds=50,
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                verbose=10,
                eval_metric="auc",
            )
        
            
        else:
            self.model.fit(
                data["train_x"][self.feature],
                data["train_y"],
                eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                eval_metric="AUC",
                verbose_eval=True,
                
            )
            



            

    def inference(self, data,save_time):
        # submission 제출하기 위한 코드
        print("### Inference && Save###")
        test_pred = self.model.predict_proba(data["test"][self.feature])[:, 1]
        data["test"]["prediction"] = test_pred
        submission = data["test"]["prediction"].reset_index(drop=True).reset_index()
        submission.rename(columns={"index": "id"}, inplace=True)
        submission_filename = f"{self.args.model}_{save_time}_submission.csv"
        submission.to_csv(
            os.path.join(self.args.output_dir, submission_filename), index=False
        )
