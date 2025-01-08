import sys
import os
from datetime import datetime
from tqdm import tqdm

import cv2
import numpy as np
import torch
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(script_dir)
os.environ["PYTHONPATH"] = (
    grandparent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
)
sys.path.append(grandparent_dir)

from stego.utils import get_transform, get_inverse_transform
from stego.data import create_cityscapes_colormap
from stego.stego import Stego


@hydra.main(
    config_path="cfg", config_name="video_inference.yaml", version_base="1.1"
)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # 初始化模型
    model = Stego.load_from_checkpoint(cfg.model_path)  # 加载你的模型
    model.eval().cuda()

    # 视频路径
    video_path = cfg.video_path
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_video_path = os.path.join(
        cfg.output_video_base_path,
        "inference_result",
        current_time + cfg.model_path.split("/")[-1] + "result.mp4",
    )

    # 读取视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 设置编码格式
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (frame_width, frame_height)
    )

    # 定义颜色映射
    cmap = create_cityscapes_colormap()

    # 修改这里，避免单次加载过多帧导致显存过大
    batch_frames = []
    pil_img_frames = []
    batch_size = cfg.batch_size
    num_count = 0

    # 使用 DataLoader 提前处理帧数据，优化内存和计算
    with tqdm(total=total_frames, desc="Processing frames", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 如果没有读取到帧，退出

            # 将 BGR 转为 RGB，方便进行推理
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转为 PIL 图像，并进行必要的变换
            pil_img = Image.fromarray(img)
            transform = get_transform(cfg.resolution, False, None)
            pil_img_transformed = transform(pil_img)
            pil_img_frames.append(pil_img)
            batch_frames.append(pil_img_transformed)

            if len(batch_frames) == batch_size:
                img_tensor = torch.stack(batch_frames).cuda()

                # 异步推理，避免等待
                with torch.no_grad():
                    # 推理得到 CRF 后处理的结果
                    code = model.get_code(img_tensor)
                    cluster_crfs, _ = model.postprocess(
                        code=code,
                        img=img_tensor,
                        use_crf_cluster=cfg.use_crf_cluster,
                        use_crf_linear=cfg.use_crf_linear,
                    )

                    for i in range(batch_size):
                        cluster_crf = cluster_crfs.cpu()[i]

                        # 将结果应用到原视频帧上，使用透明度合成
                        alpha = 0.5
                        cluster_image = cmap[cluster_crf].astype(np.uint8)
                        cluster_image_bgr = cv2.cvtColor(
                            cluster_image, cv2.COLOR_RGB2BGR
                        )
                        inverse_transform = get_inverse_transform((1080, 1920))
                        retransformed_img = inverse_transform(
                            Image.fromarray(cluster_image_bgr)
                        )
                        overlay_cluster = cv2.addWeighted(
                            cv2.cvtColor(
                                np.array(pil_img_frames[i]), cv2.COLOR_RGB2BGR
                            ),
                            1 - alpha,
                            cv2.cvtColor(
                                np.array(retransformed_img), cv2.COLOR_RGB2BGR
                            ),
                            alpha,
                            0,
                        )

                        # 将最终的帧写入输出视频
                        out.write(overlay_cluster)

                # 清空 batch_frames 准备下一批
                batch_frames = []
                pil_img_frames = []

            pbar.update(1)
            num_count += 1

    # 释放视频读取和写入对象
    cap.release()
    out.release()


if __name__ == "__main__":
    my_app()
