
import os
import argparse
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # LMCC Parameters
    parser.add_argument("--lmcc_m", type=float, default=0.2,
                        help="margin m in LMCC layer")
    parser.add_argument("--lmcc_s", type=float, default=100.0,
                        help="param s in LMCC layer")
    parser.add_argument("-ct", "--classifier_type", type=str, default="AAMC",
                        choices=["AAMC", "LMCC", "FC_CE"],
                        help="Use the AAMC (arcFace), LMCC (cosFace), or FC_CE (fc + CE loss)")

    # Groups parameters
    parser.add_argument("--M", type=int, default=20,help="_")
    parser.add_argument("--N", type=int, default=2, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10,
                        help="minimum number of images per class. Otherwise class is discarded")

    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("-ipe", "--iterations_per_epoch", type=int, default=2000, help="_")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="_")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="_")
    parser.add_argument("--epochs_num", type=int, default=500, help="_")
    parser.add_argument("--train_resize", type=int, default=(224, 224), help="_")
    parser.add_argument("--test_resize", type=int, default=256, help="_")
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--classifier_lr", type=float, default=0.01, help="_")
    parser.add_argument("-bb", "--backbone", type=str, default="EfficientNet_B0",
                        choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7"],
                        help="_")

    # Init parameters
    parser.add_argument("--resume_train", type=str, default=None, help="path with *_ckpt.pth checkpoint")
    parser.add_argument("--resume_model", type=str, default=None, help="path with *_model.pth model")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument("--num_workers", type=int, default=16, help="_")

    # Paths parameters
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of experiment. The logs will be saved in a folder with such name")
    parser.add_argument("--dataset_name", type=str, default="sf_xl",
                        choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k"], help="_")
    parser.add_argument("--train_set_path", type=str, default=None,
                        help="path to folder of train set")
    parser.add_argument("--val_set_path", type=str, default=None,
                        help="path to folder of val set")
    parser.add_argument("--test_set_path", type=str, default=None,
                        help="path to folder of test set")
    
    args = parser.parse_args()
    args.save_dir = os.path.join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    return args

