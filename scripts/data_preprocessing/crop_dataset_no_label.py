#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Cropped dataset generation script
#
# This script five-crops images from the input dataset and
# creates a new dataset with the generated crops.
# In STEGO, cropping (five-crop) should be performed before KNN generation.
# Hence, the images should first be cropped with this script,
# then KNNs should be generated for the new dataset.
#
# Expected input structure:
# DATA_DIR
# |-- INPUT_NAME
#     |-- imgs
#         |-- train
#         |-- val
#     |-- labels
#         |-- train
#         |-- val
#
############################################


import os
from os.path import join
from tqdm import tqdm
from torchvision.transforms.functional import five_crop

from preprocessing_utils import create_dataset_structure
from PIL import Image


DATA_DIR = "/home/tipriest/data/TerrainSeg/hit_grass"
INPUT_NAME = "VID_20220502_135318/2_preprocessed"
OUTPUT_NAME = "VID_20220502_135318/3_cropped"

# An image of size HxW will be five-cropped with target size of
# (CROP_RATIO*H)x(CROP_RATIO*W)
CROP_RATIO = 0.5
# File extension of images (in the imgs directory)
IMAGE_EXT = ".jpg"


def save_five_crop(input_name, output_dir, sample_name, file_ext):
    output_names = [
        join(output_dir, sample_name + "_" + str(i) + file_ext)
        for i in range(5)
    ]
    all_exist = True
    for name in output_names:
        if not os.path.isfile(name):
            all_exist = False
    if all_exist:
        return
    image = Image.open(input_name)
    crops = five_crop(
        image, (CROP_RATIO * image.height, CROP_RATIO * image.width)
    )
    for i, crop in enumerate(crops):
        name = output_names[i]
        if not os.path.isfile(name):
            crop.save(name)


def preprocess_samples(input_dir, output_dir, subset, input_subset):
    print(f"Processing subset {subset}")
    img_names = os.listdir(join(input_dir, "imgs", input_subset))
    for img_name in tqdm(img_names):
        sample_name = img_name.split(".")[0]
        img_path = join(
            input_dir, "imgs", input_subset, sample_name + IMAGE_EXT
        )
        # label_path = join(input_dir, "labels", input_subset, img_name)
        save_five_crop(
            img_path, join(output_dir, "imgs", subset), sample_name, IMAGE_EXT
        )
        # save_five_crop(
        #     label_path, join(output_dir, "labels", subset), sample_name, ".png"
        # )


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    # TODO:如果遇到了之前的文件夹，删掉
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", "train")
    preprocess_samples(input_dir, output_dir, "val", "val")


if __name__ == "__main__":
    main()
