import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    ### 중요 ###
    parser.add_argument("--model", default="CAT", type=str, help="model type")
    parser.add_argument('--fe', default="Y") # train시 valid set 쓸건지 안쓸건지 안 쓸꺼면 --fe N
    parser.add_argument('--trials', type=int, default=1) #랜덤 조합으로 몇번
    

    ## EDA 바뀔시 ##
    parser.add_argument(
        "--file_name", default="FE_v5.csv", type=str, help="train file name"
    )
    parser.add_argument("--cat_feats", default=['userID','testId','KnowledgeTag',
                                                'Month','DayOfWeek','TimeOfDay',
                                                'categorize_solvingTime','categorize_ProblemAnswerRate','categorize_TagAnswerRate','categorize_TestAnswerRate'

    ], nargs="+", help='categorical_feature option')


    ## 일반 ##
    parser.add_argument("--n_window", default=1, type=int, help='num val window')
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
    
    
    args = parser.parse_args()

    return args
