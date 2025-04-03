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
import scipy.io as sio


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
    return frames


def convert_all(input_dir: Path, output_file: Path):
    input_files = list(
        sorted(input_dir.glob("*.dat"), key=lambda x: int(x.stem.split("_")[-1]))
    )

    # Convert the files
    # Multi-threading
    res = thread_map(convert, input_files, total=len(input_files), disable=True)

    # # Single-threading
    # res = [convert(file) for file in tqdm(input_files)]

    frames = np.stack(res, axis=0)
    frames = np.concatenate(frames, axis=0)
    frames = np.packbits(frames, axis=2, bitorder="little")
    np.save(output_file, frames)


if __name__ == "__main__":
    proj_data_dir = Path("/media/knocking/Seagate_Basic/calib_spike")
    data_dir = proj_data_dir / "raw_data"
    frame_dir = data_dir.parent / "planes"
    # data_patch = "real_data/20241016"
    data_patch = ""
    indata_dir = data_dir / data_patch
    outdata_dir = frame_dir / data_patch
    outdata_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(indata_dir.iterdir())
    for scene in tqdm(scenes, total=len(scenes)):
        output_file = outdata_dir / (scene.stem + ".npy")
        convert_all(scene, output_file)
