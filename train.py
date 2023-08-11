
import sys
import torch
import logging
import torchmetrics
from tqdm import tqdm
import torchvision.transforms as T
from torch import optim

import test
import util
import models
import parser
import commons
from datasets import TrainDataset, TestDataset
from classifiers import AAMC, LMCC, LinearLayer

args = parser.parse_arguments()
assert args.train_set_path is not None, 'you must specify the train set path'
assert args.val_set_path is not None, 'you must specify the val set path'
assert args.test_set_path is not None, 'you must specify the test set path'

commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

#### Datasets & DataLoaders
train_augmentation = T.Compose([
        T.ToTensor(),
        T.Resize(args.train_resize),
        T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

groups = [TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class,
                       transform=train_augmentation
                       ) for n in range(args.N * args.N)]

val_dataset = TestDataset(args.val_set_path, M=args.M, N=args.N, image_size=args.test_resize)
test_dataset = TestDataset(args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)
val_dl = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
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

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")
logging.info(f"Feature dim: {model.feature_dim}")
logging.info(f"resume_model: {args.resume_model}")

if args.resume_model is not None:
    model, classifiers = util.resume_model(args, model, classifiers)

cross_entropy_loss = torch.nn.CrossEntropyLoss()

#### OPTIMIZER & SCHEDULER
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True)

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
        util.resume_train_with_groups(args, args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = start_epoch_num = 0
    best_loss = 100

scaler = torch.cuda.amp.GradScaler()
for epoch_num in range(start_epoch_num, args.epochs_num):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break
    train_acc = torchmetrics.Accuracy().to(args.device)
    train_loss = torchmetrics.MeanMetric().to(args.device)

    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % len(classifiers)
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)

    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    #### Train
    dataloader_iterator = iter(dataloader)
    model = model.train()

    tqdm_bar = tqdm(range(args.iterations_per_epoch), ncols=100, desc="")
    for iteration in tqdm_bar:
        images, labels, _ = next(dataloader_iterator)
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        with torch.cuda.amp.autocast():
            descriptors = model(images)
            # 1) 'output' is respectively the angular or cosine margin, of the AMCC or LMCC.
            # 2) 'logits' are the logits obtained multiplying the embedding for the
            # AMCC/LMCC weights. They are used to compute tha accuracy on the train batches 
            output, logits = classifiers[current_group_num](descriptors, labels)
            loss = cross_entropy_loss(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(classifiers_optimizers[current_group_num])
        scaler.update()

        train_acc.update(logits, labels)
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        del loss, images, output
        _ = tqdm_bar.refresh()

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")

    #### Validation
    val_lr_str = test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))

    train_acc = train_acc.compute() * 100
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False

    logging.info(f"E{epoch_num: 3d}, train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{args.scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}, " +
                 f"classifier_lr: {round(classifiers_optimizers[current_group_num].param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch_num: 3d}, Val LR: {val_lr_str}")

    scheduler.step(train_loss)
    util.save_checkpoint({"epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "args": args,
        "best_train_loss": best_loss
    }, is_best, args.save_dir)
    torch.cuda.empty_cache()

test_lr_str = test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))
logging.info(f"Test LR: {test_lr_str}")
