import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="/data/ephemeral/home/level2-dkt-recsys-06/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="../submit/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    # submission 파일
    parser.add_argument(
        "--submission_name",
        default="dkt_submission.csv",
        type=str,
        help="submission file name",
    )

    ### feature engineering
    # 순서: 기존 범주형 + 새로운 범주형 + 새로운 수치형

    # base : 무조건 들어갈 애들
    parser.add_argument("--base_num_feats", nargs="+", default=[])
    parser.add_argument(
        "--base_cat_feats",
        nargs="+",
        default=["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"],
    )

    # 실험 대상
    parser.add_argument("--new_num_feats", nargs="+", default=[])
    parser.add_argument("--new_cat_feats", nargs="+", default=[])

    args = parser.parse_args()

    args.num_feats = args.base_num_feats + args.new_num_feats
    args.cat_feats = args.base_cat_feats + args.new_cat_feats

    args.feats = args.cat_feats + args.num_feats

    return args
