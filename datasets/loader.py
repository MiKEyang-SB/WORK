import random
from typing import List, Dict, Tuple, Union, Iterator

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def build_dataloader(dataset, collate_fn, is_train: bool, opts, batch_size=None):

    if batch_size is None:
        batch_size = opts.TRAIN.train_batch_size if is_train else opts.TRAIN.val_batch_size #100

    if opts.local_rank == -1:
        if is_train:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)#打乱数据，适合训练
        else:
            sampler = SequentialSampler(dataset)#不打乱，确保验证集有序

        size = 1 if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size #当有多个GPU时，可以将batch_size成倍增加，适用于DataParallel

    else:#DDP
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), 
            shuffle=is_train
        )#将数据切成world_size份，每个进程取一部分
        pre_epoch = sampler.set_epoch #分布式训练的函数，需要在每个epoch开始前调用，单机训练可以忽略

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=opts.TRAIN.n_workers,
        pin_memory=opts.TRAIN.pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 if opts.TRAIN.n_workers > 0 else None,
    )

    return loader, pre_epoch