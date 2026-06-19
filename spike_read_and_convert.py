"""
Using the data from the 400*250 camera, we convert the .dat files to .npy files.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import thread_map, process_map
from tqdm import tqdm
import itertools
# import scipy.io as sio

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from spike import TFI, TFP
import cv2
from glob import glob
import matplotlib.pyplot as plt
import h5py
from time import time


def convert(input_file):
    with open(input_file, "rb") as file:
        data = np.fromfile(file, dtype=np.uint8)
    frames = data.reshape(400, 52 * 250).copy()
    frames = frames.reshape(400, 52, 250, order="F").copy()
    frames = frames[:, 0:50, :]
    frames = frames.reshape(400, 12500, order="F").copy()
    frames = np.unpackbits(frames, axis=1, bitorder="little").astype(np.uint8)
    frames = frames.reshape(400, 250, 400)
    # rotate 180
    frames = np.rot90(frames, 2, (1, 2))
    # print(f"Converted {input_file.name}: {frames.shape}, {frames.dtype}, max: {frames.max()}, min: {frames.min()}")
    return frames


def reconstruct(data: np.ndarray, save_path: Path, save_h5_name: str = None):
    # print(f"reconstructing {plane_file}")
    # data = np.load(plane_file)
    print(f"data loaded, shape: {data.shape}")
    # data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    # print(f"data unpacked, shape: {data.shape}")

    # Set beta to increase the brightness of the image
    alpha = 10.0
    beta = 20

    # Streaming TFI
    tfi = TFI(device="cuda")
    for idx, frame in tfi.reconstruct_stream(data, save_path=save_path):
        # print(f"frame {idx}, shape: {frame.shape}")
        # cv2.imshow(f"frame", frame)
        # cv2.waitKey(1)
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        cv2.imwrite(str(save_path / f"{idx}.png"), frame)

    # # TFI
    # tfi = TFI(device="cuda")
    # frames = tfi.reconstruct_stream(data)
    # print(f"frames reconstructed, shape: {frames.shape}")
    # # print(frames.shape, frames.dtype, frames.max(), frames.min()) # (6000, 250, 400) float32 1.0 0.0015600624
    # # print(frames[0, :5, :])
    # frames = (frames * 255).astype(np.uint8)

    # ### Save frames to h5 file
    # if save_h5_name is not None:
    #     print(f"saving frames to h5 file: {save_path / save_h5_name}")
    #     with h5py.File(save_path / save_h5_name, "w") as f:
    #         f.create_dataset("frames", data=frames[:, :, ::-1])  # Remember to flip the frame horizontally

    # ### Save frames to png
    # min_file_idx = 9999999999999
    # max_file_idx = -9999999999
    # # Get start_idx: get the max file index from save_path
    # # if no files in save_path, start_idx is 0
    # if len(list(save_path.glob("*.png"))) == 0:
    #     start_idx = 0
    # else:
    #     start_idx = max(int(f.stem) for f in save_path.glob("*.png")) + 1
    # print(f"start_idx: {start_idx}")
    # for idx, frame in enumerate(tqdm(frames, desc="saving TFI", leave=False)):
    #     # flip the frame horizontally
    #     frame = cv2.flip(frame, 1)
    #     # use cv2 to show the frame in float32 and gray scale
    #     frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    #     # put a text on the frame
    #     # cv2.putText(_f, f"{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #     # cv2.imshow("frame_TFI", _f)
    #     # cv2.imshow("frame_TFI", frame)
    #     cv2.imwrite(str(save_path / f"{idx + start_idx}.png"), frame)
    #     min_file_idx = min(min_file_idx, idx + start_idx)
    #     max_file_idx = max(max_file_idx, idx + start_idx)
    #     # cv2.waitKey(1)
    # # cv2.destroyAllWindows()

    # # print(f"min_file_idx: {min_file_idx}, max_file_idx: {max_file_idx}")

def convert_all(input_dir: Path, output_dir: Path):
    # print(f"Converting {input_dir} to {output_file}")
    input_files = list(
        sorted(input_dir.glob("*.dat"), key=lambda x: int(x.stem.split("_")[-1]))
    )
    # print(input_files[:5])

    # Convert the files
    # Multi-threading
    # res = thread_map(convert, input_files, total=len(input_files), disable=True)
    # print(f"Converted {len(input_files)} files.")
    res = [convert(file) for file in tqdm(input_files, desc="Converting files")]
    # res = []
    # for file in tqdm(input_files, desc="Converting files"):
    #     res.append(convert(file))

    # # Single-threading
    # res = [convert(file) for file in tqdm(input_files)]

    frames = np.stack(res, axis=0)
    frames = np.concatenate(frames, axis=0)
    # frames = np.packbits(frames, axis=2, bitorder="little")
    # np.save(output_file, frames)

    # save_path = Path("./reconstruct_TFI")
    # save_path.mkdir(parents=True, exist_ok=True)
    reconstruct(frames, output_dir, save_h5_name=f"{output_dir.name}.h5")


if __name__ == "__main__":
    proj_data_dir = Path("F:\\20250804_data\\calib_spike_250804_seqs")
    data_dir = proj_data_dir / "raw_data"
    frame_dir = data_dir.parent / "planes"
    # data_patch = "real_data/20241016"
    data_patch = ""
    indata_dir = data_dir / data_patch
    # outdata_dir = frame_dir / data_patch
    # outdata_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(indata_dir.iterdir())
    for scene in tqdm(scenes, total=len(scenes)):
        # output_file = Path(f"{scene.stem}.npy")
        output_dir = Path(f"./reconstruct_TFI/{scene.stem}")
        output_dir.mkdir(parents=True, exist_ok=True)
        convert_all(scene, output_dir)
