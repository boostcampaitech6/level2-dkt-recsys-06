import argparse

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
    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--output_dir", default="cat_submit/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )
    parser.add_argument(
        "--model_dir", default="model/", type=str, help="model directory"
    )
    
    args = parser.parse_args()

    return args
