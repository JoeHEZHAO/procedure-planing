import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    data_path = "datasets/COIN_assets"

    parser.add_argument(
        "--transformer",
        type=bool,
        default=True,
        help="default setting for transformer or RNN",
    )
    parser.add_argument(
        "--data_path", type=str, default=data_path, help="default data path"
    )
    parser.add_argument(
        "--beam_opt",
        type=int,
        default=1,
        help="chose options for beam search algrotihms. Opt 1 is looping through all actions; Opt2 is sampling",
    )
    parser.add_argument(
        "--pred_horz", type=int, default=3, help="prediction horizontal"
    )
    parser.add_argument(
        "--n_train", type=int, default=30, help="videos per task for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("-q", type=float, default=0.7,
                        help="regularization parameter")
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of dataloader workers"
    )
    parser.add_argument(
        "--use_related",
        type=int,
        default=0,
        help="1 for using related tasks during training, 0 for using primary tasks only",
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-d",
        type=int,
        default=3200,
        help="dimension of feature vector",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1e4,
        help="penalty coefficient for temporal cosntraints. Put 0 to use no temporal constraints during training",
    )
    parser.add_argument(
        "--share",
        type=str,
        default="words",
        help="Level of sharing between tasks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch Size for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="crosstask",
        help="Used dataset name for logging",
    )
    parser.add_argument(
        "--dataloader-type",
        type=str,
        default="ddn",
        help="The type of dataset processing loader: either ddn or plate",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="ddn",
        help="The type of dataset processing loader: either ddn or plate",
    )

    parser.add_argument(
        "--modeltype",
        type=str,
        default="transformer",
        help="The type of model",
    )

    parser.add_argument(
        "--spec-note",
        type=str,
        default="completeLoss",
        help="Any speical notes for experimental runs",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="The dimension of intermediate feature of transformers (or other model)",
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=64,
        help="The dimension of noise signal of GAN training/stochastics",
    )
    parser.add_argument(
        "--nlayer",
        type=int,
        default=2,
        help="The number of attention layers of Transformer model",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="The number of heads of each attention layer",
    )

    args = parser.parse_args()
    return args
