import argparse
import glob
import wandb
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data
from torch.utils.data import DataLoader
from loss import create_criterion
from sklearn.metrics import f1_score
from tqdm import tqdm
from weight import my_weight, ins_weight


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    # model save dir
    save_dir = increment_path(os.path.join(model_dir, args.name))
    print(save_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: Train_dataset
    dataset = dataset_module(
        data_dir=data_dir
    )
    num_classes = dataset.num_classes
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = data.random_split(dataset, [n_train, n_val])

    # augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: CustomAugmentation
    train_transform = transform_module(
        need='train',
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    val_transform = transform_module(
        need='val',
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    train_set.dataset.set_transform(train_transform)
    val_set.dataset.set_transform(val_transform)

    # dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)

    # weight
    weight = ins_weight(train_set.dataset, device, dataset.num_classes)

    # loss
    criterion = create_criterion(args.criterion, weight=weight)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    # logging
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer
    }
    wandb.init(project=args.name, entity='danielkim30433', config=config)
    # train
    wandb.watch(model)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_total = 0
        matches = 0
        f1_total = 0
        for idx, train_batch in enumerate(tqdm(train_loader)):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)  # don't input the preds

            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            match = (preds == labels).sum().item()
            matches += match
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            f1_total += f1
            wandb.log({"train batch loss": loss.item(),
                       "train batch f1": f1,
                       "train batch acc": match / args.batch_size})
        wandb.log({"train loss": loss_total / len(train_loader),
                   "train f1": f1_total / len(train_loader),
                   "train acc": matches / args.batch_size / len(train_loader)
                   })

        scheduler.step()

        # val loop
        with torch.no_grad():
            val_loss_toal = 0
            val_f1_total = 0
            val_matches = 0
            model.eval()
            for val_batch in tqdm(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                val_loss_toal += criterion(outs, labels).item()
                val_matches += (labels == preds).sum().item()
                val_f1_total += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            val_acc = val_matches / args.valid_batch_size / len(val_loader)
            val_f1 = val_f1_total / len(val_loader)
            val_loss = val_loss_toal / len(val_loader)
            wandb.log({"val loss": val_loss,
                       "val f1": val_f1,
                       "val acc": val_acc})
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:.5f}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{save_dir}/last.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=55, help='random seed (default: 55)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 50)')
    parser.add_argument('--dataset', type=str, default='TrainDataset', help='dataset type (default: TrainDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation',
                        help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[300, 300],
                        help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Resnet18', help='model type (default: Resnet18)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20,
    #                    help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
