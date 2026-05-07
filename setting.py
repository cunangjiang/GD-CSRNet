import torch
import torch.distributed as dist
from collections.abc import Sequence
import math
import numpy as np

def initialize_distributed(num_gpus: int) -> tuple:
    """
    Initialize distributed training.

    Returns:
        tuple: local_rank, world_size, and device.
    """
    
    # all_gpus = list(range(torch.cuda.device_count()))  # 获取所有可用GPU索引
    # if 0 in all_gpus:
    #     all_gpus.remove(0)  # 移除第0块GPU
    # assert len(all_gpus) >= num_gpus, "Not enough GPUs available after excluding GPU 0."

    # # 设置 CUDA_VISIBLE_DEVICES 环境变量
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, all_gpus[:num_gpus]))
    if torch.cuda.is_available() and num_gpus > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return local_rank, world_size, device

def partition_dataset(
    data,
    num_partitions,
    even_divisible,
    shuffle,
    ratios=None,
    seed=0,
    drop_last=False,
):
    """
    Split the dataset into N partitions. It can support shuffle based on specified random seed.
    Will return a set of datasets, every dataset contains 1 partition of original dataset.
    And it can split the dataset based on specified ratios or evenly split into `num_partitions`.
    Refer to: https://pytorch.org/docs/stable/distributed.html#module-torch.distributed.launch.

    Note:
        It also can be used to partition dataset for ranks in distributed training.
        For example, partition dataset before training and use `CacheDataset`, every rank trains with its own data.
        It can avoid duplicated caching content in each rank, but will not do global shuffle before every epoch:

        .. code-block:: python

            data_partition = partition_dataset(
                data=train_files,
                num_partitions=dist.get_world_size(),
                shuffle=True,
                even_divisible=True,
            )[dist.get_rank()]

            train_ds = SmartCacheDataset(
                data=data_partition,
                transform=train_transforms,
                replace_rate=0.2,
                cache_num=15,
            )

    Args:
        data: input dataset to split, expect a list of data.
        ratios: a list of ratio number to split the dataset, like [8, 1, 1].
        num_partitions: expected number of the partitions to evenly split, only works when `ratios` not specified.
        shuffle: whether to shuffle the original dataset before splitting.
        seed: random seed to shuffle the dataset, only works when `shuffle` is True.
        drop_last: only works when `even_divisible` is False and no ratios specified.
            if True, will drop the tail of the data to make it evenly divisible across partitions.
            if False, will add extra indices to make the data evenly divisible across partitions.
        even_divisible: if True, guarantee every partition has same length.

    Examples::

        >>> data = [1, 2, 3, 4, 5]
        >>> partition_dataset(data, ratios=[0.6, 0.2, 0.2], shuffle=False)
        [[1, 2, 3], [4], [5]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False)
        [[1, 3, 5], [2, 4]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=True)
        [[1, 3], [2, 4]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=True, drop_last=False)
        [[1, 3, 5], [2, 4, 1]]
        >>> partition_dataset(data, num_partitions=2, shuffle=False, even_divisible=False, drop_last=False)
        [[1, 3, 5], [2, 4]]

    """
    data_len = len(data)
    datasets = []

    indices = list(range(data_len))
    if shuffle:
        # deterministically shuffle based on fixed seed for every process
        rs = np.random.RandomState(seed)
        rs.shuffle(indices)

    if ratios:
        next_idx = 0
        rsum = sum(ratios)
        for r in ratios:
            start_idx = next_idx
            next_idx = min(start_idx + int(r / rsum * data_len + 0.5), data_len)
            datasets.append([data[i] for i in indices[start_idx:next_idx]])
        return datasets

    if not num_partitions:
        raise ValueError("must specify number of partitions or ratios.")
    # evenly split the data without ratios
    if not even_divisible and drop_last:
        raise RuntimeError("drop_last only works when even_divisible is True.")
    if data_len < num_partitions:
        raise RuntimeError(f"there is no enough data to be split into {num_partitions} partitions.")

    if drop_last and data_len % num_partitions != 0:
        # split to nearest available length that is evenly divisible
        num_samples = math.ceil((data_len - num_partitions) / num_partitions)
    else:
        num_samples = math.ceil(data_len / num_partitions)
    # use original data length if not even divisible
    total_size = num_samples * num_partitions if even_divisible else data_len

    if not drop_last and total_size - data_len > 0:
        # add extra samples to make it evenly divisible
        indices += indices[: (total_size - data_len)]
    else:
        # remove tail of data to make it evenly divisible
        indices = indices[:total_size]

    for i in range(num_partitions):
        _indices = indices[i:total_size:num_partitions]
        datasets.append([data[j] for j in _indices])

    return datasets
