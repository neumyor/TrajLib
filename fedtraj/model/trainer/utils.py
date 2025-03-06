import time
import logging
import pickle
import torch
import torch.nn as nn
import heapq
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from fedtraj.utils.traj import *
from fedtraj.utils.data_loader import TrajDataset


def collate_and_augment(trajs, cellspace, embs, augfn1, augfn2):
    # trajs: list of [[lon, lat], [,], ...]

    # 1. augment the input traj in order to form 2 augmented traj views
    # 2. convert augmented trajs to the trajs based on mercator space by cells
    # 3. read cell embeddings and form batch tensors (sort, pad)

    trajs1 = [augfn1(t) for t in trajs]
    trajs2 = [augfn2(t) for t in trajs]

    trajs1_cell, trajs1_p = zip(*[merc2cell2(t, cellspace) for t in trajs1])
    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs2])

    trajs1_emb_p = [
        torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs1_p
    ]
    trajs2_emb_p = [
        torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p
    ]

    trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first=False).to(Config.device)
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False).to(Config.device)

    trajs1_emb_cell = [embs[list(t)] for t in trajs1_cell]
    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]

    trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first=False).to(
        Config.device
    )  # [seq_len, batch_size, emb_dim]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False).to(
        Config.device
    )  # [seq_len, batch_size, emb_dim]

    trajs1_len = torch.tensor(
        list(map(len, trajs1_cell)), dtype=torch.long, device=Config.device
    )
    trajs2_len = torch.tensor(
        list(map(len, trajs2_cell)), dtype=torch.long, device=Config.device
    )

    # return: two padded tensors and their lengths
    return (
        trajs1_emb_cell,
        trajs1_emb_p,
        trajs1_len,
        trajs2_emb_cell,
        trajs2_emb_p,
        trajs2_len,
    )


def collate_for_test(trajs, cellspace, embs):
    # trajs: list of [[lon, lat], [,], ...]

    # behavior is similar to collate_and_augment, but no augmentation

    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs])
    trajs2_emb_p = [
        torch.tensor(generate_spatial_features(t, cellspace)) for t in trajs2_p
    ]
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False).to(Config.device)

    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False).to(
        Config.device
    )  # [seq_len, batch_size, emb_dim]

    trajs2_len = torch.tensor(
        list(map(len, trajs2_cell)), dtype=torch.long, device=Config.device
    )
    # return: padded tensor and their length
    return trajs2_emb_cell, trajs2_emb_p, trajs2_len


@torch.no_grad()
def lcss_test():
    logging.info("[Lcss_Test]start.")
    import traj_dist.distance as tdist
    from tqdm import tqdm

    # varying downsampling; varying distort
    vt = Config.test_type
    results = []
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        with open(
            Config.dataset_file + "_newsimi_" + vt + "_" + str(rate) + ".pkl", "rb"
        ) as fh:
            q_lst, db_lst = pickle.load(fh)
            querys, databases = q_lst, db_lst
            n_querys = len(querys)
            n_databases = len(databases)
            dists = np.zeros((n_querys, n_databases))
            for i, query in tqdm(
                enumerate(querys), desc=f"start {vt} lcss count", unit="trajs"
            ):
                for j, database in enumerate(databases):
                    dists[i, j] = tdist.lcss(np.array(query), np.array(database))
            targets = np.diag(dists)  # [1000]
            rank = np.sum(
                np.sum(dists[:, :n_databases] <= targets[:, np.newaxis], axis=1)
            ) / len(q_lst)
            results.append(rank)
    logging.info(
        "[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}".format(
            vt, *results
        )
    )


def get_federated_segments(traj_dataset: TrajDataset, fed_num=1):
    logging.info(f"[Split dataset and convert to segments] start converting")
    dataset_split = tool_funcs.split(traj_dataset, fed_num)
    dataset_split = [TrajDataset(cut_trajectorys_into_segments(dataset.data)) for dataset in dataset_split]
    logging.info(f"[Split dataset and convert to segments] done")

    return dataset_split


def cut_trajectorys_into_segments(df):
    from fedtraj.utils.trajclus import traclus_partition
    import pandas as pd
    from tqdm import tqdm

    new_data = []
    for index, row in tqdm(df.iterrows()):
        # 读取merc_seq和wgs_seq
        merc_seq = row["merc_seq"]
        wgs_seq = row["wgs_seq"]

        # 调用traj_clus函数得到切分点布尔数组
        _, split_points = traclus_partition(merc_seq)

        # 找到所有切分点的索引
        split_indices = np.where(split_points)[0]

        # 处理没有切分点的情况
        if len(split_indices) == 0:
            new_trajlen = len(merc_seq)
            new_merc_seq = merc_seq
            new_wgs_seq = wgs_seq
            new_data.append([new_trajlen, new_wgs_seq, new_merc_seq])
        else:
            # 切分轨迹
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1]
                new_trajlen = end - start + 1
                new_merc_seq = merc_seq[start : end + 1]
                new_wgs_seq = wgs_seq[start : end + 1]
                new_data.append([new_trajlen, new_wgs_seq, new_merc_seq])

    # 创建新的DataFrame
    new_df = pd.DataFrame(new_data, columns=["trajlen", "wgs_seq", "merc_seq"])
    return new_df
