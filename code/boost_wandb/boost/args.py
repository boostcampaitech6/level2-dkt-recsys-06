import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="model/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="submit/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    ### boosting model 관련 argument ###
    parser.add_argument("--n_estimators", default=100, type=int, help="n_estimators")
    parser.add_argument("--random_state", default=42, type=int, help="random_state")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument("--max_depth", default=6, type=int, help="max_depth")
    parser.add_argument("--reg_lambda", default=0.0, type=float, help="reg_lambda")
    parser.add_argument("--subsample", default=1.0, type=float, help="subsample")
    parser.add_argument("--early_stopping_rounds", default=10, type=int, help="early_stopping_rounds")
    parser.add_argument("--max_leaves", default=31, type=int, help="max_leaves")
    
    ### lgbm & catboost
    parser.add_argument("--min_data_in_leaf", default=20, type=int, help="min_data_in_leaf")
    
    ### lgbm
    parser.add_argument("--feature_fraction", default=1.0, type=float, help="feature_fraction")
    parser.add_argument("--bagging_freq", default=0, type=int, help="bagging_freq")

    ### xgboost


    ### catboost


    ### 중요 ###
    parser.add_argument("--model", default="CAT", type=str, help="model type")
    parser.add_argument('--feats', nargs="+", default=[], help="추가할 feature")
    parser.add_argument('--trials', default=2, type=int) # 랜덤 조합으로 몇번
    parser.add_argument('--cat_feats', nargs='+', default=[], help='범주형 feature')
    
    args = parser.parse_args()
    args.feats = args.feats[0].split(' ')

    return args
