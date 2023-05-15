import os
from os.path import join
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as plt

from stego.utils import *
from stego.stego import STEGO
from stego.data import ContrastiveSegDataset

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="cfg", config_name="eval_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = os.path.join(cfg.output_root, cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)

    model = STEGO.load_from_checkpoint(cfg.model_path)
    test_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        image_set="val",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type=model.backbone_name,
        resolution=cfg.resolution
    )

    test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                            shuffle=True, num_workers=cfg.num_workers,
                            pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()

    for i, batch in enumerate(tqdm(test_loader)):
        if i >= cfg.n_batches:
            break
        with torch.no_grad():
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            cluster_preds, linear_preds = model(img)

            model.linear_metrics.update(linear_preds, label)
            model.cluster_metrics.update(cluster_preds, label)
        

    tb_metrics = {
        **model.linear_metrics.compute(),
        **model.cluster_metrics.compute(),
    }
    print(tb_metrics)

if __name__ == "__main__":
    prep_args()
    my_app()