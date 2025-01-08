import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torchvision

script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(script_dir)
os.environ["PYTHONPATH"] = (
    grandparent_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
)
sys.path.append(grandparent_dir)

from stego.utils import get_transform, get_inverse_transform, get_pure_transform
from stego.data import create_cityscapes_colormap
from stego.stego import Stego

@hydra.main(config_path="cfg", config_name="video_inference.yaml", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # 初始化模型
    model = Stego.load_from_checkpoint(cfg.model_path)  # 加载你的模型
    model.eval().cuda()

    # 视频路径
    video_path = cfg.video_path
    output_video_path = cfg.output_video_path

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
    num_count = 0
    # 创建 tqdm 进度条
    with tqdm(total=total_frames, desc="Processing frames", ncols=100) as pbar:
        # 逐帧读取视频
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or num_count > 250:
                break  # 如果没有读取到帧，退出
            num_count += 1
            if num_count % 5 != 0:
                continue
            # 将 BGR 转为 RGB，方便进行推理
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转为 PIL 图像，并进行必要的变换
            pil_img = Image.fromarray(img)
            transform = get_transform(cfg.resolution, False, None)
            pil_img_transformed = transform(pil_img)
            img_tensor = pil_img_transformed.unsqueeze(0).cuda()

            ####
            # pil_img.save(
            #     os.path.join(
            #         f"/home/tipriest/Pictures/hit_grass/pil_img_{num_count}.png"
            #     )
            # )
            transform2 = get_pure_transform(cfg.resolution)
            pil_img_transform2 = transform2(pil_img)
            # pil_img_transform2.save(
            #     os.path.join(
            #         f"/home/tipriest/Pictures/hit_grass/pil_img_transform2_{num_count}.png"
            #     )
            # )

            with torch.no_grad():
                # 推理得到 CRF 后处理的结果
                code = model.get_code(img_tensor)
                cluster_crf, _ = model.postprocess(
                    code=code,
                    img=img_tensor,
                    use_crf_cluster=True,
                    use_crf_linear=True,
                )

                # 将 CRF 结果转换为 NumPy 数组
                cluster_crf = cluster_crf.cpu()[0]
                # 将结果应用到原视频帧上，使用透明度合成
                alpha = 0.5  # 透明度
                cluster_image = cmap[cluster_crf].astype(np.uint8)
                # cluster_image.show()
                cluster_image_bgr = cv2.cvtColor(
                    cluster_image, cv2.COLOR_RGB2BGR
                )
                # Image.fromarray(cluster_image).save(
                #     os.path.join("/home/tipriest/Pictures/testQ2E3D.png")
                # )
                # out.write(cluster_image_bgr)
                # cv2.imshow("Frame with Overlay", frame)

                ####
                # inverse_transform = get_inverse_transform((1080, 1920))
                # retransformed_img = inverse_transform(pil_img_transform2)
                # retransformed_img.save(
                #     os.path.join(
                #         f"/home/tipriest/Pictures/hit_grass/retransformed_img_{num_count}.png"
                #     )
                # )
                inverse_transform = get_inverse_transform((1080, 1920))
                retransformed_img = inverse_transform(
                    Image.fromarray(cluster_image_bgr)
                )
                # retransformed_img.save(
                #     os.path.join(f"/home/tipriest/Pictures/FECewfwef_{num_count}.png")
                # )
                retransformed_img_bgr = cv2.cvtColor(
                    np.array(retransformed_img), cv2.COLOR_RGB2BGR
                )

                overlay_cluster = cv2.addWeighted(
                    cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR),
                    1 - alpha,
                    cv2.cvtColor(
                        np.array(retransformed_img), cv2.COLOR_RGB2BGR
                    ),
                    alpha,
                    0,
                )
                cv2.imwrite(
                    f"/home/tipriest/Pictures/hit_grass/overlay_cluster_{num_count}.png",
                    overlay_cluster,
                )

                # out.write(retransformed_img_bgr)
                # overlay_cluster = cv2.addWeighted(
                #     frame, 1 - alpha, retransformed_img_bgr, alpha, 0
                # )
                # out.write(overlay_cluster)

                # 按 'q' 键退出
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                # 将 CRF 结果图像转为 BGR 格式
                #

                # cv2.imshow("pic_test", cluster_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # 通过透明度将推理结果覆盖到原图像
                #

                # final_frame = overlay_cluster
                # # 将最终的帧写入输出视频
                # out.write(final_frame)
                pbar.update(1)
    # 释放视频读取和写入对象
    cap.release()
    out.release()


if __name__ == "__main__":
    my_app()
