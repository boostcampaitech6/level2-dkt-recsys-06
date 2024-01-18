import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    ### 중요 ###
    parser.add_argument("--model", default="CAT", type=str, help="model type")
    parser.add_argument('--fe', default="N") # train시 valid set 쓸건지 안쓸건지
    parser.add_argument('--trials', type=int, default=50) #랜덤 조합으로 몇번
    

    ## 일반 ##
    parser.add_argument(
        "--file_name", default="FE.csv", type=str, help="train file name"
    )
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
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")

    parser.add_argument("--cat_feats", default=[], nargs="+", help='categorical_feature option')
    
    args = parser.parse_args()

    return args
