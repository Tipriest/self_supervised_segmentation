#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Freiburg Forest preprocessing script
#
# http://deepscene.cs.uni-freiburg.de/
# Abhinav Valada, Gabriel L. Oliveira, Thomas Brox, Wolfram Burgard
# Deep Multispectral Semantic Scene Understanding of Forested Environments using Multimodal Fusion
# International Symposium on Experimental Robotics (ISER), Tokyo, Japan, 2016.
#
#
# Download Friburg Forest:
# wget http://deepscene.cs.uni-freiburg.de/static/datasets/download_freiburg_forest_annotated.sh
# bash download_freiburg_forest_annotated.sh
# tar -xzf freiburg_forest_annotated.tar.gz
# rm freiburg_forest_annotated.tar.gz*
#
#
#
# Expected input structure after unpacking in DATA_DIR:
# DATA_DIR
# |-- INPUT_NAME
#     |-- train
#         |-- rgb
#         |-- GT_color
#     |-- test
#         |-- rgb
#         |-- GT_color
#
# Output structure after preprocessing:
# DATA_DIR
# |-- OUTPUT_NAME
#     |-- imgs
#         |-- train
#         |-- val
#     |-- labels
#         |-- train
#         |-- val
############################################


import os
from os.path import join
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil



DATA_DIR = "/home/tipriest/data/TerrainSeg/hit_grass/"
INPUT_NAME = "Videos_Annotated/VID_20220502_135318"
OUTPUT_NAME = "Videost_preprocessed/VID_20220502_135318"

FF_CMAP = np.array(
    [
        (0, 0, 0),  # Object
        (170, 170, 170),  # Road
        (0, 255, 0),  # Grass
        (102, 102, 51),  # Vegetation
        (0, 120, 255),  # Sky
        (0, 60, 0,),    # Tree
        # (separate color present in the dataset, 
        # but assigned to class Vegetation in the dataset's official readme)
    ]
)

def create_dataset_structure(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs"), exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs", "train"), exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs", "val"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels", "val"), exist_ok=True)


def convert_rgb_label(label, cmap):
    for i in range(cmap.shape[0]):
        color = cmap[i]
        indices = np.all(label == color, axis=2)
        label[indices] = i
    return np.unique(label, axis=-1).squeeze()




def preprocess_and_copy_image(input_name, output_name, is_label=False, rgb_label=False, cmap=None):
    """
        如果是rgb label的话,按照cmap制造出类别映射后的image
        否则，单纯地拷贝图片到指定的文件夹
    """
    if os.path.isfile(output_name):
        return
    if is_label and rgb_label:
        if cmap is None:
            raise ValueError("No colormap provided to convert the RGB label")
        image = Image.open(input_name).convert("RGB")
        img = np.array(image)
        img = convert_rgb_label(img, cmap)
        image = Image.fromarray(img)
        image.save(output_name)
    else:
        shutil.copyfile(input_name, output_name)


def preprocess_and_copy_label(input_name, output_name, cmap):
    """
        相当于preprocess_and_copy_image的简化版, 直接拷贝对应的label
    """
    if os.path.isfile(output_name):
        return
    image = Image.open(input_name).convert("RGB")
    img = np.array(image)
    img = convert_rgb_label(img, cmap)
    img[img == 5] = 3  # Class Tree assigned to Vegetation
    image = Image.fromarray(img)
    image.save(output_name)


def preprocess_samples(input_dir, output_dir, input_subset, output_subset):
    print("Processing subset {}".format(output_subset))
    label_names = os.listdir(join(input_dir, input_subset, "rgb"))
    for label_name in tqdm(label_names):
        sample_name = label_name.split("_")[0]
        img_path = join(input_dir, input_subset, "rgb", sample_name + "_Clipped.jpg")
        label_path = join(input_dir, input_subset, "GT_color", label_name)
        preprocess_and_copy_image(input_name=img_path,
                                  output_name=join(output_dir, "imgs", output_subset, sample_name + ".jpg"),
                                  is_label=False)
        preprocess_and_copy_label(
            label_path,
            os.path.join(output_dir, "labels", output_subset, sample_name + ".png"),
            FF_CMAP,
        )


def main():
    input_dir = os.path.join(DATA_DIR, INPUT_NAME)
    output_dir = os.path.join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, input_subset="train", output_subset="train")
    preprocess_samples(input_dir, output_dir, input_subset="test", output_subset="val")


if __name__ == "__main__":
    main()
