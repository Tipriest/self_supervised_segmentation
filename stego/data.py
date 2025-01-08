#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import os
import random
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from stego.utils import get_nn_file_name


class UnlabeledImageFolder(Dataset):
    """
    A simple Dataset class to read images from a given folder.
    """

    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = root
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.images[index])).convert(
            "RGB"
        )
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)

def colors_hexcode2colors_tuple(colors_hexcode):
    colors = []
    for color in colors_hexcode:
        # 将十六进制颜色码转换为 RGB 元组
        r = (color >> 16) & 0xFF  # 提取红色分量
        g = (color >> 8) & 0xFF   # 提取绿色分量
        b = color & 0xFF          # 提取蓝色分量
        # 结果为十进制格式，因此不需要进一步转换
        colors.append((r, g, b))
    return colors

        
def create_cityscapes_colormap():
    colors_hexcode = [
        0x804080,
        0xfaaa1e,
        0xdcdc00,
        0x98fb98,
        0x00005a,
        0x966464,
        0xF423E8,
        0xFAAAA0,
        0x464646,
        0x66669c,
        0xbe9999,
        0xb4a5b4,
        0x96785a,
        0x999999,
        0x6b8e23,
        0x4682b4,
        0xdc143c,
        0xff0000,
        0x00008e,
        0xe6968c,
        0x000046,
        0x003c64,
        0x00006e,
        0x005064,
        0x0000e6,
        0x770b20,
        0x000000,
    ]
    colors = colors_hexcode2colors_tuple(colors_hexcode)
    return np.array(colors)


class DirectoryDataset(Dataset):
    """
    A Dataset class that reads images and (if available) labels from the given directory.
    The expected structure of the directory:
    data_dir
    |-- dataset_name
        |-- imgs
            |-- image_set
        |-- labels
            |-- image_set

    If available, file names in labels/image_set should be the same as file names
    in imgs/image_set (excluding extensions).

    If labels are not available (there is no labels folder) this class returns
    zero arrays of shape corresponding to the image shape.
    """

    def __init__(
        self, data_dir, dataset_name, image_set, transform, target_transform
    ):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        self.dataset_name = dataset_name
        self.dir = os.path.join(data_dir, dataset_name)
        self.img_dir = os.path.join(self.dir, "imgs", self.split)
        self.label_dir = os.path.join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert (
            len(self.img_files) > 0
        ), f"Could not find any images in dataset directory {self.img_dir}"
        if os.path.exists(os.path.join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(
                self.label_files
            ), f"The {self.dataset_name} dataset contains a different number\
                of images and labels:\
                    {len(self.img_files)} images and {len(self.label_files)} labels"
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_name = self.img_files[index]
        img = Image.open(os.path.join(self.img_dir, image_name))
        if self.label_files is not None:
            label_name = self.label_files[index]
            label = Image.open(os.path.join(self.label_dir, label_name))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)
        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64)

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class ContrastiveSegDataset(Dataset):
    """
    The main Dataset class used by STEGO.
    Internally uses the DirectoryDataset class to load images.
    Additionally, this class uses the precomputed Nearest Neighbor files
    to extract the knn corresponding image for STEGO training.
    It returns a dictionary containing an image and its positive pair
    (one of the nearest neighbor images).
    STEGO使用的主要的数据集格式
    内部使用DirectoryDataset类来加载图片
    除此之外, 这个类使用预先计算的最近邻图片来为STEGO的训练提供KNN相关图片
    它能够返回一个包含图片本身和它的最相似图片的文件夹
    """

    def __init__(
        self,
        data_dir,
        dataset_name,
        nn_file_dir,
        image_set,
        transform,
        target_transform,
        model_type,
        resolution,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        num_neighbors=5, #TODO: 这里看下这个num_neighbors的定义
        mask=False,
        pos_labels=False,
        pos_images=False,
        extra_transform=None,
    ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform
        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = DirectoryDataset(
            data_dir, dataset_name, image_set, transform, target_transform
        )

        feature_cache_file = get_nn_file_name(
            nn_file_dir=nn_file_dir,
            model_type=model_type,
            image_set=image_set,
            resolution=resolution,
        )
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file):
                raise ValueError(
                    f"could not find nn file {feature_cache_file} please run precompute_knns"
                )
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert (
                len(self.dataset) == self.nns.shape[0]
            ), "Found different numbers of images in\
                dataset {dataset_name} and nn file {feature_cache_file}"

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][
                torch.randint(
                    low=1, high=self.num_neighbors + 1, size=[]
                ).item()
            ]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        coord_entries = torch.meshgrid(
            [
                torch.linspace(-1, 1, pack[0].shape[1]),
                torch.linspace(-1, 1, pack[0].shape[2]),
            ]
        )
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
        }

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        if self.aug_photometric_transform is not None:
            img_aug = self.aug_photometric_transform(
                self.aug_geometric_transform(pack[0])
            )

            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)

            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret
