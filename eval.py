
import sys
import torch
import logging

import test
import util
import models
import parser
import commons
from classifiers import AAMC, LMCC, LinearLayer
from datasets import initialize, TrainDataset, TestDataset


args = parser.parse_arguments()
assert args.train_set_path is not None, 'you must specify the train set path'
assert args.test_set_path is not None, 'you must specify the test set path'

commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

#### Datasets & DataLoaders
groups = [TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class
                       ) for n in range(args.N * args.N)]

test_dataset = TestDataset(args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

#### Model
model = models.GeoClassNet(args.backbone).to(args.device)

# Each group has its own classifier, which depends on the number of classes in the group
if args.classifier_type == "AAMC":
    classifiers = [AAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "LMCC":
    classifiers = [LMCC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "FC_CE":
    classifiers = [LinearLayer(model.feature_dim, group.get_classes_num()) for group in groups]

logging.info(f"Feature dim: {model.feature_dim}")

if args.resume_model is not None:
    model, classifiers = util.resume_model(args, model, classifiers)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

lr_str = test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))
logging.info(f"LR: {lr_str}")
