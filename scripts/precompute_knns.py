#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# KNN computation for datasets used for training STEGO
#
# This script generates the KNN file for a new dataset to be used with STEGO.
# Before running the script, preprocess the dataset (including cropping).
# Adjust the path to the dataset,
# subsets to be processed and target resolution in cfg/knn_config.yaml
#
############################################

import os
import sys
from os.path import join

# from typing import Any
import hydra
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
from datetime import datetime


grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = (
    grandparent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
)
sys.path.append(grandparent_dir)

from stego.data import ContrastiveSegDataset
from stego.stego import Stego
from stego.utils import prep_args, get_transform, get_nn_file_name


def get_feats(model, loader):
    """_summary_

    Args:
        model (_type_): 神经网络模型，用于从输入图像中提取特征
        loader (_type_): 数据加载器（如 DataLoader), 用于提供批次数据

    Returns:
        _type_: _description_
    """
    all_feats = []
    # pack是一个包含一个批次数据的字典
    for pack in tqdm(loader):
        # 假设 pack 是一个字典，并且包含键 "img"，
        # 该行从字典中提取图像数据（即一个批次的图像）
        # img 的形状为 (batch_size, channels, height, width)
        # 在这里是(96, 3, 224, 224)
        img = pack["img"]
        img = img.cuda()
        # feats是模型对这一批图像前向推理之后的结果，
        # feats是一个元组，有两个元素
        # 第一个元素的shape是(96, 768, 28, 28)
        # 第二个元素的shape是(96, 90, 28, 28)
        feats = model.forward(img)
        feats = feats[0]
        # 对特征图的空间维度（height 和 width）进行均值池化。
        # 这将把每个图像的特征图从 (batch_size, channels, height, width) 降维为
        # (batch_size, channels)，即每个通道的均值特征。
        # 均值之后的shape为(96, 768)
        feats = feats.mean([2, 3])
        # 对每个通道的特征做归一化
        feats = F.normalize(feats, dim=1)
        # feats = F.normalize(model.forward(img.cuda())[0].mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="cfg", config_name="knn_config.yaml", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0)

    # 获取当前时间，格式化为 yyyy-mm-dd_hh-mm
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    nns_save_path = join(cfg.data_dir, "nns", current_time)
    os.makedirs(nns_save_path)

    image_sets = cfg.image_sets

    res = cfg.resolution
    n_batches = 16
    model = Stego(1).cuda()

    for image_set in image_sets:
        feature_cache_file = get_nn_file_name(
            nn_file_dir=nns_save_path,
            model_type=model.backbone.backbone_type,
            image_set=image_set,
            resolution=res,
        )
        if not os.path.exists(feature_cache_file):
            print(f"{feature_cache_file} not found, computing")
            dataset = ContrastiveSegDataset(
                data_dir=cfg.data_dir,
                dataset_name=cfg.dataset_name,
                nn_file_dir=nns_save_path,
                image_set=image_set,
                transform=get_transform(res, False, "center"),
                target_transform=get_transform(res, True, "center"),
                model_type=model.backbone.backbone_type,
                resolution=res,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )

            with torch.no_grad():
                normed_feats = get_feats(model, loader)
                all_nns = []
                # 批次特征被分成了16份
                step = normed_feats.shape[0] // n_batches
                print(f"normed_feats's shape is {normed_feats.shape}")
                for i in tqdm(range(0, normed_feats.shape[0], step)):
                    torch.cuda.empty_cache()
                    batch_feats = normed_feats[i : i + step, :]
                    # torch.einsum 是 PyTorch 的一个非常强大的函数，
                    # 它允许我们通过爱因斯坦求和约定来进行张量运算。
                    # 这里的 "nf,mf->nm" 表示：
                    # batch_feats 是一个形状为 (step, d) 的张量，
                    # d 是特征维度，对应爱因斯坦和, n 是批次大小（step），f 是特征维度。
                    # normed_feats 是一个形状为 (N, d) 的张量，
                    # N 是总样本数，f 也是特征维度。
                    pairwise_sims = torch.einsum(
                        "nf,mf->nm", batch_feats, normed_feats
                    )
                    # 返回的是 pairwise_sims 中每行的前30个最大值及其对应的索引。
                    # [1] 选择的是索引部分（即前30个相似度最大值的索引），
                    # 这些索引代表了每个样本与其他样本的前30个最近邻。
                    nns = torch.topk(pairwise_sims, cfg.num_neighbors)
                    nns = nns[1]
                    all_nns.append(nns)
                    del pairwise_sims
                # 在所有批次处理完成后，
                # 使用 torch.cat 将 all_nns 中的所有最近邻索引按第0维（即行）拼接起来，
                # 得到最终的 nearest_neighbors 张量。
                # nearest_neighbors 是一个形状为 (N, 30) 的张量,
                # 其中每行包含了每个样本的前30个最近邻的索引。
                nearest_neighbors = torch.cat(all_nns, dim=0)

                np.savez_compressed(
                    feature_cache_file, nns=nearest_neighbors.numpy()
                )
                print(
                    "Saved NNs",
                    model.backbone.backbone_type,
                    cfg.dataset_name,
                    image_set,
                )


if __name__ == "__main__":
    prep_args()
    my_app()
