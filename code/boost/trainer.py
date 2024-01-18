import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
import lightgbm as lgbm
import optuna
import joblib
import json

# optuna
def objective(trial,args, FEATURE,data):
    
    # CAT : regressor
    if args.model == "CAT":
        params_CAT = {
        'has_time' : True,        
        'random_seed': 42,
        'objective': 'Logloss',  # 이진 분류 문제
        'custom_metric': 'accuracy',  # 평가 지표로 AUC 사용
        'eval_metric': 'AUC',  # AUC를 사용하여 평가
      
        'iterations': trial.suggest_int('iterations', 1000, 5000), # iterations: 트리의 개수 또는 부스팅 라운드 수
        'od_wait':trial.suggest_int('od_wait', 1000, 2000), #'od_wait': 최적화 중단 기다리는 라운드 수
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # 학습 속도'depth': trial.suggest_int('depth', 6, 10), # 'depth':각 트리의 최대 깊이를 나타냅니다. 트리의 깊이가 클수록 모델은 더 복잡한 관계를 학습
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 12), # 리프 노드에 필요한 최소 데이터 포인트 수를 나타냅니다. 작은 값은 더 많은 리프를 생성하고 모델을 더 복잡하게 만듬
        'leaf_estimation_iterations': trial.suggest_categorical('leaf_estimation_iterations',[1,5,10,13]), # leaf_estimation_iterations: 리프 가중치를 추정하기 위한 반복 횟수
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',  0.001 ,1), # L2 정규화 항의 강도
        'border_count': trial.suggest_int('border_count', 32, 100), # 이진 트리를 만들 때 사용되는 특징의 최대 수입니다. 주어진 범위 내에서 정수로 추정
        'random_strength': trial.suggest_int('random_strength', 1, 20), # 'random_strength': 트리에서 특징을 무작위로 선택하는 강도입니다. 주어진 범위 내에서 부동 소수점 값
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 0.5), # 'bagging_temperature': 부스팅 트리의 각 반복에서 샘플을 선택하는 온도 매개변수입니다. 주어진 범위 내에서 부동 소수점 값
        }

        bst = CatBoostClassifier(**params_CAT,task_type='GPU', devices='cuda',verbose=100)
        bst.fit(data["train_x"][FEATURE], data["train_y"], cat_features= args.cat_feats, eval_set=(data["valid_x"][FEATURE], data["valid_y"]))

        # 예측
        # y_pred_proba = bst.predict(data["valid_x"][FEATURE]) #회귀
        y_pred_proba = bst.predict_proba(data["valid_x"][FEATURE])[:, 1] #분류
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_proba]

    # XG : classifier
    elif args.model == "XG":
        param_XG = {
        'random_state': 42,
        'objective': 'binary:logistic',  # 이진 분류 문제
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',  # GPU 가속화를 사용하려면 'gpu_hist'로 설정
        'booster': 'gbtree',  # 일반적인 그래디언트 부스팅 결정 
        'num_round' : trial.suggest_int('num_round', 1000, 5000),  

        'tree_method': 'hist',  # 또는 'gpu_hist'로 설정
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
        # 데이터셋
        train_data = xgb.DMatrix(data["train_x"][FEATURE], label=data["train_y"])
        valid_data = xgb.DMatrix(data["valid_x"][FEATURE], label=data["valid_y"])

        # xgboost 모델 훈련
        bst = xgb.train(param_XG, train_data, evals=[(valid_data, 'validation')], verbose_eval=100)
        
        # 예측
        dtest = xgb.DMatrix(data["valid_x"][FEATURE])
        y_pred_proba = bst.predict(dtest)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_proba]

    # LGBM : classifier
    elif args.model == "LGBM":
        params_LGBM = {
        'random_state': 42,
        'objective': 'binary',  # 이진 분류 문제
        'metric': 'auc',  # 평가 지표로 AUC 사용
        'boosting_type': 'gbdt',  # gbdt는 일반적인 그래디언트 부스팅 결정 
        'num_round' : trial.suggest_int('num_round', 1000, 5000),  
        
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),  # 트리의 최대 리프 노드 개수
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),  # 학습 속도
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),  # 각 트리에 사용할 특성의 비율
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),  # 리프 노드에 필요한 최소 데이터 수
        'max_depth': trial.suggest_int('max_depth', 3, 15),  # 트리의 최대 깊이
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),  # L1 정규화 강도
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),  # L2 정규화 강도
        'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0),  # 분할 최소 이득
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 1e+2),  # 자식 노드에서 필요한 최소 가중치 합계
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),  # 각 트리에 사용할 데이터의 비율
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),  # 데이터 샘플링 빈도
        'max_bin': trial.suggest_int('max_bin', 32, 512),  # 히스토그램 분할 중 최대 bin 수
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),  # 양성 클래스의 가중치
        'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 10.0),  # 카테고리 특징을 부드럽게 하는 파라미터
        }

        # 데이터셋
        train_data = lgbm.Dataset(data["train_x"][FEATURE], label=data["train_y"])
        valid_data = lgbm.Dataset(data["valid_x"][FEATURE], label=data["valid_y"], reference=train_data)
    
        # LightGBM 모델 훈련
        bst = lgbm.train(params_LGBM,train_data, valid_sets=valid_data, categorical_feature=args.cat_feats)

        # 예측
        y_pred_proba = bst.predict(data["valid_x"][FEATURE])
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_proba]

    
    # 정확도 및 AUC 계산
    accuracy = accuracy_score(data["valid_y"], y_pred_binary)
    auc = roc_auc_score(data["valid_y"], y_pred_proba)
    print('Accuracy: {:.4f}'.format(accuracy))
    print('AUC: {:.4f}'.format(auc))

    return auc  # auc 최대화하는 방향으로


class boosting_model:
    def __init__(self, args,FEATURE,data):
        self.args = args
        self.feature = FEATURE
        self.data = data
        
        if args.model == "CAT":
            # Optuna 최적화
            study = optuna.create_study(direction='maximize',study_name='CatBoostRegressor',sampler=optuna.samplers.TPESampler(seed=self.args.seed)) #seed args에서 끌어오기
            study.optimize(lambda trial: objective(trial,self.args,self.feature, self.data), n_trials=args.trials)

            # 최적 하이퍼파라미터 출력
            print('Hyperparameters: {}'.format(study.best_params))
            
            self.model = CatBoostClassifier(
                **study.best_params,task_type='GPU', devices='cuda',  
                custom_metric = 'accuracy', eval_metric = 'AUC',
                objective= 'Logloss'              
            )
            
        elif args.model == "XG":
            # Optuna 최적화
            study = optuna.create_study(direction='maximize', study_name='XGBoostRegressor',sampler=optuna.sampler.TPESampler(seed=self.args.seed))
            study.optimize(lambda trial: objective(trial,self.args,self.feature, self.data), n_trials=args.trials)

            # 최적 하이퍼파라미터 출력
            print('Hyperparameters: {}'.format(study.best_params))

            self.model = xgb.XGBClassifier(
                **study.best_params, objective= 'binary:logistic', eval_metric= 'auc',
            )

        elif args.model == "LGBM":
            # Optuna 최적화
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial,self.args,self.feature, self.data), n_trials=args.trials)

            # 최적 하이퍼파라미터 출력
            print('Hyperparameters: {}'.format(study.best_params))
            
            self.model = lgbm.LGBMClassifier(
               **study.best_params, objective = 'binary', metric = 'auc'
            )

        else:
            raise Exception("cat,xg,lgbm 중 하나의 모델을 선택해주세요")
        

    def training(self, data, args, FEATURE,FE_train):
        # CAT
        if args.model == "CAT":
            if args.fe == "N":
                self.model.fit(
                    FE_train[FEATURE],
                    FE_train["answerCode"],
                    cat_features= self.args.cat_feats,
                    verbose=200,
                    )
            else:
                print("Valid Data is used while training")
                self.model.fit(
                    data["train_x"][self.feature],
                    data["train_y"],
                    eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                    cat_features= self.args.cat_feats,
                    verbose=200,
                    )
            print(self.model.get_best_score())
            print(self.model.get_all_params())
            model_type =  'catboost'

        # XG
        elif args.model == "XG":
            if args.fe == "N":
                self.model.fit(
                    FE_train[FEATURE],
                    FE_train["answerCode"],
                    verbose=200,
                    )
            else:
                print("Valid Data is used while training")
                self.model.fit(
                    data["train_x"][self.feature],
                    data["train_y"],
                    eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                    verbose=200,
                    )
            model_type = 'xgboost'  

        # LGBM
        else:
            if args.fe == "N":
                self.model.fit(
                    FE_train[FEATURE],
                    FE_train["answerCode"],
                    )
            else:
                print("Valid Data is used while training")
                self.model.fit(
                    data["train_x"][self.feature],
                    data["train_y"],
                    eval_set=[(data["valid_x"][self.feature], data["valid_y"])],
                    )
            
            model_type = 'lightgbm' 
        
        get_feature_importance(self.model, FEATURE, model_type)
            
    
            

    def inference(self, data,save_time):
        # submission 제출하기 위한 코드
        #test_pred = self.model.predict_proba(data["test"][self.feature])[:, 1]
        test_pred = self.model.predict(data["test"][self.feature])
        data["test"]["prediction"] = test_pred
        submission = data["test"]["prediction"].reset_index(drop=True).reset_index()
        submission.rename(columns={"index": "id"}, inplace=True)
        submission_filename = f"{self.args.model}_{save_time}_submission.csv"
        submission.to_csv(
            os.path.join(self.args.output_dir, submission_filename), index=False
        )


def get_feature_importance(model, feature_names, model_type):
        if model_type ==  'catboost':
            importance = model.get_feature_importance()
        elif model_type == 'xgboost':
            importance = model.feature_importances_
        elif model_type == 'lightgbm':
            importance = model.feature_importances_

        feature_importance = np.array(importance)
        feature_names = np.array(feature_names)
    
        # DataFrame 생성하고 정렬
        fi_df = pd.DataFrame({'Feature Names': feature_names, 'Feature Importance': feature_importance})
        fi_df = fi_df.sort_values(by='Feature Importance', ascending=False)

        # Print the results
        print(f"{model_type} Feature Importance:")
        print(fi_df)
        