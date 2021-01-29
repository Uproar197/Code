import optuna
import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler

# my configuration file
from config import config   #
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, model, writer_dict, device):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    # x = 0
    for i_iter, batch in enumerate(trainloader):
        # print("====================")
        images, labels, _, _ = batch
        # print('------ image size: {}'.format(images.shape))
        images = images.to(device)
        labels = labels.long().to(device)

        losses, _ = model(images, labels)  # 调用utils.py 中的FullModel类的Forward方法
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        # ave_loss.update(reduced_loss.item())
        ave_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.TRAIN.PRINT_FREQ == 0 and rank == 0:  # config.PRINT_FREQ=20
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), lr, print_loss)
            logging.info(msg)


def validate(config, testloader, model, writer_dict, device):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()  # 定义一个该类对象
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, pred = model(image, label)  #
            pred = F.upsample(input=pred, size=(
                size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)  # 获取混淆矩阵

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)
    confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average() / world_size
    return print_loss, mean_IoU, IoU_array


def objective(trial):

    # Ｉ used 5 GPUs.
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))
    if distributed:
        torch.cuda.set_device(args.local_rank)
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK  # True
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # False
    cudnn.enabled = config.CUDNN.ENABLED  # True

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)


    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(  # eval()在哪定义的？
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,  # 只使用部分数据进行训练时使用
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)  # 调用datasets下对应的数据集

    if distributed:
        train_sampler = DistributedSampler(train_dataset)   # torch自带函数
    else:
        train_sampler = None

###########################
    # validate_set is gotten using similar method.

    # loss function, myself's function.
    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                             weight=train_dataset.class_weights)


    model = FullModel(model, criterion)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-3)


    optimizer = torch.optim.SGD([{'params':
                                      filter(lambda p: p.requires_grad,
                                             model.parameters()),
                                  'lr': lr}],
                                lr=lr,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=wd,
                                nesterov=config.TRAIN.NESTEROV,
                                )

    epoch_iters = np.int(train_dataset.__len__() /
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    best_index = 0
    start = timeit.default_timer()
    #end_epoch = trial.suggest_int('end_epoch', 4, 6)
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = end_epoch * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # training function
        train(config, epoch, end_epoch,
              epoch_iters, lr, num_iters,
              trainloader, optimizer, model, writer_dict,
              device)
        # validate function
        valid_loss, mean_IoU, IoU_array = validate(config,
                    testloader, model, writer_dict, device)
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            best_index = epoch
            torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'best_model.pth'))
            print('======> Save the best model  ---> best_model.pth !!!!!')
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

        trial.report(mean_IoU, epoch)
        print('=======>used trial report !!!')
        if trial.should_prune():
            print('----> using trial prune !!!')
            raise optuna.TrialPruned()
    return mean_IoU


print('========> Now, creating STUDY !!!')

mypruner = optuna.pruners.NopPruner()
mysampler = optuna.samplers.TPESampler(n_startup_trials=3)
study = optuna.create_study(study_name='main_study', sampler=mysampler, storage='sqlite:///main_study_250.db',
        direction="maximize", load_if_exists=True, pruner=mypruner)
print('***********************************************************')
study.optimize(objective, n_trials=500)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# 获取每个epoch的最好超参数
# trials = study.trials
# best_params_for_each_epochs = []
# for i in range(n_epochs):
#     j = np.min([trial.intermediate_values[i] for trial in trials])
#     best_params_for_each_epochs.append(trials[j].params)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(" Best trial:  Value: ", trial.value)
print("Best trial: number {}".format(trial.number))

print(" Best trial:  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

