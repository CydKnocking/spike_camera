import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


class CLASS_RECON:
    def __init__(self, device="cpu", planesPerFrame=10, interval=10):
        self.device = torch.device(device)
        self.planesPerFrame = planesPerFrame
        self.interval = interval

    def post_process(self, frames):
        # Rescale the frames to 0-1\
        # frames = np.clip(frames, 1 / 30, 1 / 3)
        # show_hist(1 / frames[10])
        frames = frames / frames.max()
        frames = frames / np.median(frames) * 0.3
        frames = frames.clip(0, 1)
        frames = frames ** (1 / 2.2)
        return frames

    def make_video(self, frames: torch.Tensor, video_path: Path, fps=30.0):
        # 定义视频文件的保存路径和参数
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = frames[0].shape

        # 创建视频写入器
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, fps, (width, height), isColor=False
        )

        # 输出第一帧的图像
        cv2.imwrite(video_path.with_suffix(".png"), frames[0])
        # 将每一帧写入视频文件
        for frame in frames:
            video_writer.write(frame)

        # 释放视频写入器
        video_writer.release()

    def reconstruct(self, planes, if_new_planes=False):
        all_frames = []
        new_planes = [] if self.__class__.__name__ == "TFI_DN" else None
        for idx in tqdm(range(0, planes.shape[0], 2000), desc="Reconstruct", leave=False):
            if self.__class__.__name__ == "TFI_DN":
                x, y = self._reconstruct(planes[idx : idx + 2000])
                all_frames.append(x)
                new_planes.append(y)
            else:
                all_frames.append(self._reconstruct(planes[idx : idx + 2000]))
        if if_new_planes:
            return np.concatenate(all_frames, axis=0), np.concatenate(
                new_planes, axis=0
            )
        else:
            return np.concatenate(all_frames, axis=0)


class TFI(CLASS_RECON):
    def _reconstruct(self, planes):
        planes = torch.tensor(planes, dtype=torch.int16).to(self.device)
        num_planes, height, width = planes.shape
        left = torch.zeros((height, width), dtype=torch.int16).to(self.device)
        right = torch.zeros((num_planes + 1, height, width), dtype=torch.int16).to(
            self.device
        )
        frames = torch.zeros(
            (num_planes // self.interval, height, width), dtype=torch.float32
        ).to(self.device)

        spike = torch.zeros((height, width), dtype=torch.float32).to(self.device)
        left = torch.zeros((height, width), dtype=torch.float32).to(self.device)

        for i in range(num_planes - 1, 0, -1):
            right[i] = right[i + 1] + 1
            if i < num_planes - 1:
                right[i][planes[i + 1] == 1] = 1

        for i in range(num_planes):
            left += 1
            ones = planes[i] == 1
            zeros = planes[i] == 0
            if i % self.interval == self.interval - 1:
                spike[ones] = left[ones]
                spike[zeros] = left[zeros] + right[i][zeros]
                frames[i // self.interval] = spike.float()

            left[ones] = 0
        frames = 1.0 / (frames)
        planes = planes.cpu().numpy()
        return frames.cpu().numpy()


class TFP(CLASS_RECON):
    def _reconstruct(self, planes):
        planes = torch.tensor(planes, dtype=torch.int16).to(self.device)
        num_planes, height, width = planes.shape

        frames = torch.zeros(
            ((num_planes - self.planesPerFrame) // self.interval + 1, height, width),
            dtype=torch.float32,
        ).to(device=self.device)

        id = 0
        for i in range(self.planesPerFrame, num_planes + 1, self.interval):
            frames[id] = torch.sum(planes[i - self.planesPerFrame : i], dim=0)
            id += 1
        frames = frames / self.planesPerFrame
        return frames.cpu().numpy()