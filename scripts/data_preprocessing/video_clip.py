import os
from os.path import join
import shutil
from datetime import datetime, date, time
import cv2
import ffmpeg
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

VIDEO_PATH = "/home/tipriest/data/TerrainSeg/hit_grass/Videos/VID_20220502_135318.mp4"
VIDEO_CLIPPED_FRAMES_PATH = "/home/tipriest/data/TerrainSeg/hit_grass/Videos_ClippedFrames"
VIDEO_CLIPPED_FRAMES_PATH = join(VIDEO_CLIPPED_FRAMES_PATH, os.path.splitext(os.path.basename(VIDEO_PATH))[0])


def get_video_parameters(video_path):
    """
    使用 OpenCV 获取视频参数
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的主要参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate  # 总时长（秒）

    cap.release()

    # 输出视频参数
    print(f"视频分辨率: {width} x {height}")
    print(f"帧率: {frame_rate} 帧/秒")
    print(f"总帧数: {total_frames}")
    print(f"视频时长: {duration:.2f} 秒")

    return width, height, frame_rate, total_frames, duration


def get_video_metadata(video_path):
    """
    get video parameter
    """

    # 使用 ffmpeg 获取更多元数据（比如编码格式、比特率等）
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=codec_name,bit_rate')
        codec_name = probe['streams'][0]['codec_name']
        bit_rate = int(probe['streams'][0]['bit_rate']) / 1e6  # 转换为 Mbps
        print(f"视频编码格式: {codec_name}")
        print(f"比特率: {bit_rate:.2f} Mbps")
    except ffmpeg.Error as e:
        print(f"无法获取视频元数据: {e}")


def calculate_ssim(frame1, frame2):
    """计算两帧之间的结构相似性(SSIM)"""
    # 转为灰度图像进行比较
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


def calculate_histogram_diff(frame1, frame2):
    """计算两帧之间的直方图差异"""
    # 转换为HSV色彩空间
    frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # 计算直方图并进行归一化
    hist1 = cv2.calcHist([frame1_hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2_hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # 计算直方图的相关性
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def extract_frames(video_path, output_folder, step = 8, use_contrast=False, ssim_threshold=0.9, hist_threshold=0.95):
    """
    从视频中提取每一帧输出到指定的文件夹中
    """
    # 检查文件夹是否存在，如果存在则删除，然后重新创建文件夹
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"旧文件夹 {output_folder} 已删除.")
    os.makedirs(output_folder)
    print(f"文件夹 {output_folder} 已创建.")

    # 使用 OpenCV 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频共有 {total_frames} 帧，帧率为 {frame_rate} 帧/秒.")
    previous_frame = None
    frame_count = 0
    pic_name_count = 0
    step_cur = 0

    for frame_count in tqdm(range(total_frames), desc="提取视频帧", unit="帧"):
        ret, frame = cap.read()
        if step_cur == step:
            step_cur = 0
            pic_name_count += 1
        else:
            step_cur += 1
            continue

        if not ret:  # 如果没有帧可读，视频已读取完
            break
        if use_contrast:
            if previous_frame is not None:
                # 计算 SSIM
                ssim_value = calculate_ssim(previous_frame, frame)
                # 计算直方图差异
                hist_diff = calculate_histogram_diff(previous_frame, frame)

                # 如果两帧相似，跳过当前帧
                if ssim_value > ssim_threshold and hist_diff > hist_threshold:
                    # print(f"frame_{frame_count:04d}.jpg's ssim_value = {ssim_value}, hist_diff = {hist_diff}")
                    continue
        # 生成输出图片的文件路径
        frame_filename = os.path.join(output_folder, f"frame_{pic_name_count:04d}.jpg")

        # 保存当前帧为图片
        cv2.imwrite(frame_filename, frame)
        previous_frame = frame  # 更新上一帧

        frame_count += 1

    # 释放视频对象
    cap.release()
    print("所有帧已提取完成.")


if __name__ == "__main__":
    time1 = datetime.now()
    print(time1.strftime("开始时间: %Y年%m月%d日 %H: %M:%S %p"))
    print("使用 OpenCV 获取视频参数:")
    get_video_parameters(VIDEO_PATH)
    print("\n使用 FFmpeg 获取视频元数据:")
    get_video_metadata(VIDEO_PATH)
    extract_frames(VIDEO_PATH, VIDEO_CLIPPED_FRAMES_PATH)
    time2 = datetime.now()
    print(time2.strftime("结束时间: %Y年%m月%d日 %H: %M:%S %p"))
    # 计算耗时
    time_diff = time2 - time1
    hours = time_diff.seconds // 3600
    minutes = (time_diff.seconds % 3600) // 60
    seconds = time_diff.seconds % 60

    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
