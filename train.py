import argparse
import glob
import json
import os
import random
import re
import nni
import pandas as pd
import logging
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from nni.utils import merge_parameter

from dataset import TestDataset, MaskBaseDataset
from loss import create_criterion
from tqdm import tqdm
from pandas import Series, DataFrame


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, args, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """

    folder_name = str(args['classification'])+"_"+str(args['model'])+"_"+str(args['batch_size'])+"_"+str(args['criterion'])+"_"+str(args['lr'])+"_Center"
    path = Path(os.path.join(path, folder_name, args['name']))

    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def make_csv(model, data_iter, epoch, submission, save_dir, accr, loss):
    with torch.no_grad():
        model.eval()  # evaluate (affects DropOut을 안하고, and BN은 학습되어있는 것을 사용)

        hard_predictions = []
        soft_predictions = None

        for batch_in in data_iter:
            model_pred = model(batch_in)
            _, y_pred = torch.max(model_pred.data, 1)
            hard_predictions.extend(y_pred.cpu().numpy())
            soft_predictions = model_pred.data if soft_predictions is None else torch.cat(
                [soft_predictions, model_pred.data], dim=0)

        # hard submission
        submission['ans'] = hard_predictions
        submission.to_csv(os.path.join(save_dir, 'submission' + str(epoch) +"_"+str(accr)[:6]+"_"+str(loss)[:6]+ '_hard.csv'), index=False)

        # soft submission
        soft_predictions = soft_predictions.transpose(0, 1).contiguous().cpu().numpy()
        for idx in range(soft_predictions.shape[0]):
            submission[str(idx)] = soft_predictions[idx]
        submission.to_csv(os.path.join(save_dir, 'submission' + str(epoch) +"_"+str(accr)[:6]+"_"+str(loss)[:6]+ '_soft.csv'), index=False)

        print("submission.csv is generated")

        model.train()  # back to train mode


def get_num_classification(c):

    num_classes = None

    if c == "mask":
        num_classes = 3
    elif c == "gender":
        num_classes = 2
    elif c == "age":
        num_classes = 3
    else:
        num_classes = 3 * 2 * 3

    return num_classes

def train(data_dir, model_dir, args):
    seed_everything(args['seed'])

    save_dir = increment_path(model_dir, args)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args['dataset'])  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
        c=args['classification']
    )
    num_classes = get_num_classification(args['classification'])  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args['augmentation'])  # default: BaseAugmentation
    transform_train = transform_module(
        resize=args['resize'],
        mean=dataset.mean,
        std=dataset.std,
    )

    dataset.set_transform(transform_train)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()


    train_loader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        num_workers=2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args['valid_batch_size'],
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- test data iter
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    test_image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    test_set = TestDataset(test_image_paths, args['resize'])

    transform_module = getattr(import_module("dataset"), "BaseAugmentation")  # default: BaseAugmentation
    transform_test = transform_module(
        resize=args['resize'],
        mean=dataset.mean,
        std=dataset.std,
    )
    test_set.set_transform(transform_test)

    test_loader = DataLoader(
        test_set,
        batch_size=args['batch_size'],
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args['model'])  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args['criterion'], num_classes)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args['optimizer'])  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args['lr'],
        weight_decay=5e-5
    )

    # scheduler = StepLR(optimizer, args['lr_decay_step'], gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)


    best_val_acc = 0
    best_train_acc = 0
    best_val_loss = np.inf
    best_train_loss = np.inf
    n_total = 0

    for epoch in range(args['epochs']):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        n_total = 0
        for idx, train_batch in enumerate(tqdm(train_loader)):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            n_total += inputs.size(0)

        loss_avg_value = loss_value / len(train_loader)
        train_acc = matches / n_total
        current_lr = get_lr(optimizer)
        print(
            f"Epoch[{epoch}/{args['epochs']}]({idx + 1}/{len(train_loader)}) || "
            f"training loss {loss_avg_value:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
        )


        nni.report_intermediate_result(train_acc)


        if train_acc > 0.89 and best_train_loss > loss_avg_value:
            submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
            make_csv(model, test_loader, epoch, submission, save_dir, train_acc, loss_avg_value)
            best_train_loss = loss_avg_value
            best_train_acc = train_acc


        if epoch in [10, 20, 25]:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            print("Loss 1/10")

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, preds, args['dataset'] != "MaskSplitByProfileDataset")

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)

            # report intermediate result
            nni.report_intermediate_result(val_acc)
            # logger.debug('validation accuracy %g', val_acc)
            # logger.debug('Pipe send intermediate result done.')

            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()

    # report final result
    nni.report_final_result(best_train_acc)


def get_params(parser):

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 35)')
    parser.add_argument('--classification', type=str, default='multi', help='classification type (default: multi)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[160, 160],
                        help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--valid_batch_size', type=int, default=128,
                        help='input batch size for validing (default: 128)')
    parser.add_argument('--model', type=str, default='Efficientnet_b6', help='model type (default: Efficientnet_b6)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/code/baseline/model'))

    args, _ = parser.parse_known_args() # parse_args와 유사하게 동작하지만 여분의 인자에 대해 error를 발생시키지 않음
    print("args : ", args)

    return args

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()

        # get parameters form tuner
        tuner_params = nni.get_next_parameter()

        args = vars(merge_parameter(get_params(parser), tuner_params))

        # test 할 때
        # tuner_params = dict({"epochs": 10, "lr":0.01})
        # args = vars(merge_parameter(get_params(parser), tuner_params))

        print(args)

        data_dir = args['data_dir']
        model_dir = args['model_dir']

        train(data_dir, model_dir, args)

    except Exception as exception:

        raise
