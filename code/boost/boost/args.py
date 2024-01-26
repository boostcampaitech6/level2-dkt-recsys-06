import argparse

<<<<<<< HEAD
Feature = [ 'Itemseq', 'SolvingTime', 'CumulativeTime', 'UserAvgSolvingTime',
       'RelativeUserAvgSolvingTime', 'CumulativeItemCount', 'Item_last7days',
       'Item_last30days', 'CumulativeUserItemAcc', 'PastItemCount',
       'UserItemElapsed', 'UserRecentItemSolvingTime', 'ItemAcc',
       'AverageItemSolvingTime_Correct', 'AverageItemSolvingTime_Incorrect',
       'AverageItemSolvingTime', 'RelativeItemSolvingTime',
       'SolvingTimeClosenessDegree', 'UserTagAvgSolvingTime', 'TagAcc',
       #'CumulativeUserTagAverageAcc', 
       'CumulativeUserTagExponentialAverage',
       'UserTagCount', 'UserTagElapsed',  'TestAcc', ]

Categorical_Feature = ['userID', 'assessmentItemID', 'testId','KnowledgeTag',
                       'Month','DayOfWeek', 'TimeOfDay', 'WeekOfYear', 
       'UserRecentTagAnswer',
       'UserRecentItemAnswer',
       #'categorize_solvingTime',
       #'categorize_ItemAcc', 'categorize_TagAcc', 'categorize_TestAcc',
       #'categorize_CumulativeUserItemAcc',
       #'categorize_CumulativeUserTagAverageAcc',
       #'categorize_CumulativeUserTagExponentialAverage', 'CategorizedDegree'

]
Feature = Feature + Categorical_Feature

def parse_args():
    parser = argparse.ArgumentParser()
    
    ### 중요 ###
    parser.add_argument("--model", default="CAT", type=str, help="model type")
    parser.add_argument('--trials', type=int, default=10) #랜덤 조합으로 몇번
    parser.add_argument("--file_name", default="FE_v9.csv", 
                        type=str, help="train file name"
    )

    ## 일반 ##
    parser.add_argument("--feature", default=Feature,
                         nargs="+", help='categorical_feature option')
    parser.add_argument("--cat_feats", default=Categorical_Feature,
                        nargs="+", help='categorical_feature option')
    
    parser.add_argument("--n_fold", default=5, type=int, help='num val fold')
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")

    ## 경로 ##
=======

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
>>>>>>> wonhee
    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--output_dir", default="cat_submit/", type=str, help="output directory"
=======
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
>>>>>>> wonhee
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )
<<<<<<< HEAD
    parser.add_argument(
        "--model_dir", default="model/", type=str, help="model directory"
    )
    
    args = parser.parse_args()
=======

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
    
    args = parser.parse_args()
    args.feats = args.feats[0].split(' ')
>>>>>>> wonhee

    return args
