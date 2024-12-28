import os
import shutil
import random

clipped_frames_folder = '/home/tipriest/data/TerrainSeg/hit_grass/Videos_ClippedFrames/VID_20220502_135318'  # 源文件夹路径
# base annotated folder
BASE_ANNOTATED_FOLDER = '/home/tipriest/data/TerrainSeg/hit_grass/Videos_Annotated'
VIDEO_NAME = clipped_frames_folder.split('/')[-1]
BASE_ANNOTATED_FOLDER = os.path.join(BASE_ANNOTATED_FOLDER, VIDEO_NAME)

TRAIN_SUB_FOLDER = 'train/rgb'  # 训练集文件夹路径
TEST_SUB_FOLDER = 'test/rgb'  # 测试集文件夹路径
TRAIN_FOLDER = os.path.join(BASE_ANNOTATED_FOLDER, TRAIN_SUB_FOLDER)
TEST_FOLDER = os.path.join(BASE_ANNOTATED_FOLDER, TEST_SUB_FOLDER)


def split_images(src_folder, train_folder, test_folder, train_ratio=0.8):
    """
    SPLIT IMAGES
    """
    # 确保目标文件夹存在
    if os.path.exists(BASE_ANNOTATED_FOLDER):
        shutil.rmtree(BASE_ANNOTATED_FOLDER)
        print(f"旧文件夹 {BASE_ANNOTATED_FOLDER} 已删除.")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 获取源文件夹中的所有图片文件
    all_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 打乱文件列表
    random.shuffle(all_files)

    # 计算训练集和测试集的分割点
    train_size = int(len(all_files) * train_ratio)

    # 获取训练集和测试集的文件列表
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    # 将图片分配到训练集文件夹
    for file in train_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(train_folder, file)
        shutil.copy(src_path, dst_path)

    # 将图片分配到测试集文件夹
    for file in test_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(test_folder, file)
        shutil.copy(src_path, dst_path)

    print(f"总共有 {len(all_files)} 张图片，{len(train_files)} 张已分配到训练集，{len(test_files)} 张已分配到测试集。")


if __name__ == "__main__":
    # 示例使用
    split_images(clipped_frames_folder, TRAIN_FOLDER, TEST_FOLDER, train_ratio=0.8)
