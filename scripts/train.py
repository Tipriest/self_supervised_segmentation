#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# STEGO training script
#
# This script trains a new STEGO model from scratch of from a given checkpoint.
#
# Before running, adjust parameters in cfg/train_config.yaml.
#
# The hyperparameters of the model and the learning rates can be adjusted in
# stego/cfg/model_config.yaml.
#
############################################

import os
import sys
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import pytorch_lightning

# from pytorch_lightning.loggers import TensorBoardLogger

# import torch.multiprocessing
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(script_dir)
os.environ["PYTHONPATH"] = (
    grandparent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
)
sys.path.append(grandparent_dir)

from stego.stego import Stego
from stego.utils import prep_args, get_transform
from stego.data import ContrastiveSegDataset


@hydra.main(
    config_path="cfg", config_name="train_config.yaml", version_base="1.1"
)
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    checkpoint_save_path = os.path.join(
        cfg.checkpoint_dir, "checkpoint", current_time
    )
    wandb_name = cfg.wandb_name + current_time
    seed_everything(seed=0)

    if cfg.model_path:
        model = Stego.load_from_checkpoint(cfg.model_path).cuda()
    else:
        model = Stego(cfg.num_classes).cuda()

    if cfg.reset_clusters:
        model.reset_clusters(cfg.num_classes, cfg.extra_clusters)

    train_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        nn_file_dir=cfg.nn_file_dir,
        image_set="train",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type=model.backbone.backbone_type,
        resolution=cfg.resolution,
        num_neighbors=cfg.num_neighbors,
        pos_images=True,
        pos_labels=True,
    )

    val_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        nn_file_dir=cfg.nn_file_dir,
        image_set="val",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type=model.backbone.backbone_type,
        resolution=cfg.resolution,
    )

    train_loader = DataLoader(
        train_dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=wandb_name,
        log_model=cfg.wandb_log_model,
    )

    trainer = pytorch_lightning.Trainer(
        logger=wandb_logger,
        max_steps=cfg.max_steps,
        default_root_dir=checkpoint_save_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_save_path,
                every_n_train_steps=cfg.val_check_interval,
                save_top_k=2000,
                monitor="val/cluster/mIoU",
                mode="max",
            )
        ],
        # gpus=1,
        val_check_interval=cfg.val_check_interval,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
