"""
Read spike camera dat files and save as npy file.
modified by bajimh

1M, 2M are the resolution of the camera.
"""

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import time
import tyro
from tqdm.contrib.concurrent import process_map

check_tmpl = np.zeros((2000,), dtype=np.uint32)


def compose_channel_and_line(channel: int, line: int):
    assert isinstance(channel, int) and 0 <= channel < 4
    assert isinstance(line, int) and 1 <= line <= 500
    return (channel << 14) | (line << 4)


for i in range(500):
    check_tmpl[i * 2] = compose_channel_and_line(0, 500 - i)
    check_tmpl[i * 2 + 1] = compose_channel_and_line(1, 500 - i)
    check_tmpl[1000 + i * 2] = compose_channel_and_line(2, 1 + i)
    check_tmpl[1000 + i * 2 + 1] = compose_channel_and_line(3, 1 + i)


def is_valid_frame(bytes: np.ndarray):
    assert bytes.shape == (2000, 64) and bytes.dtype == np.uint8
    # print("bytes", bytes)
    check_bytes = (bytes[..., -1].astype(np.uint32) << 8) | (
        bytes[..., -2].astype(np.uint32)
    )
    check_bytes = check_bytes & 0xFFF0
    # print("check_bytes", check_bytes)
    return (check_bytes == check_tmpl).all()


def repack_frame(bytes: np.ndarray):
    assert bytes.shape == (2000, 64) and bytes.dtype == np.uint8
    bytes = bytes.reshape(1000, 1024 // 8, order="C").copy()
    frame = np.unpackbits(bytes, axis=-1, bitorder="little")
    frame = np.concatenate([frame[..., :500], frame[..., 512:-12]], axis=-1)

    # rotate 180 degree
    frame = np.rot90(frame, 2)
    return np.packbits(frame, axis=-1, bitorder="little")


def process1M(args):
    index, files, max_time = args
    with open(files[index], "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    if index > 0:
        with open(files[index - 1], "rb") as f:
            prev_data = np.fromfile(f, dtype=np.uint8)
        data = np.concatenate([prev_data[-1999 * 64 :], data])
    frames_count = 0
    frames_buffer = np.zeros((400, 1000, 1000 // 8), dtype=np.uint8)
    pointer = 0
    # print(f"Reading {index}")
    st = time.time()
    while pointer <= data.shape[0] - 2000 * 64:
        possible_frame = data[pointer : pointer + 2000 * 64]
        possible_frame = possible_frame.reshape(2000, 64, order="C").copy()
        if is_valid_frame(possible_frame):
            frames_buffer[frames_count] = repack_frame(possible_frame)
            frames_count += 1
            pointer += 2000 * 64
        else:
            pointer += 1
        if time.time() - st > max_time:
            print(f"Break {index}")
            break
    # print(f"End {index}")
    # print(frames_count, frames_buffer.shape)
    frames = frames_buffer[:frames_count]
    # print(f"Finish {index}")
    # print(frames.shape)
    # assert 1==0
    return frames


def process2M(args):
    index, files, max_time = args
    with open(files[index], "rb") as f:
        data = f.read()
        #data = np.fromfile(f, dtype=np.uint8)
    if(index > 0):
        with open(files[index - 1], "rb") as f:
            prev_data = f.read()
            #prev_data = np.fromfile(f, dtype=np.uint8)
        data = prev_data[-1920*1080//8:] + data
    frames_count = 0
    frames_buffer = np.zeros((400, 1920, 1080 // 8), dtype=np.uint8)
    frame_size = (1920 + 1920 * 1080) // 8
    pointer = 0
    # print(f"Reading {index}")
    st = time.time()
    length = len(data)
    #print(length)
    while pointer <= length - frame_size:
        possible_frame = data[pointer : pointer + frame_size]
        #print(possible_frame[:4])
        if (possible_frame[:4] == b'\xF0\xF0\xF1\xF1'):
            #possible_frame = possible_frame.reshape(2000, 64, order="C").copy()
            frames_buffer[frames_count] = np.frombuffer(possible_frame[1920//8:],dtype=np.uint8).reshape(1920, 1080 // 8, order="C")
            frames_count += 1
            pointer += frame_size
        else:
            pointer += 1
            #print(pointer)
        if time.time() - st > max_time:
            print(f"Break {index}")
            break
    # print(f"End {index}")
    frames = frames_buffer[:frames_count]
    return frames


def decompress(arr, num_bits=8):
    mask = (1 << np.arange(num_bits - 1, -1, -1)).astype(np.uint8)

    # 将每个8位元素扩展为8个单独的位
    expanded_data = (arr[..., None] & mask) != 0

    # 将二维数组转换为一维数组
    expanded_data = expanded_data.reshape(expanded_data.shape[:-2] + (-1,))

    return expanded_data.astype(np.uint8)


def prepareData(data, min_frame, max_frame):
    # real data is too large
    if min_frame >= 0:
        data = data[min_frame:max_frame].copy()
    data = data.astype(np.uint8)
    return data


def read_dat(
    input_dir: Path,
    output_file: Path,
    min_frame: int = -1,
    max_frame: int = -1,
    max_time: int = 60,
    data_type: str = "1M",
):
    """Read spike camera dat files and save as npy file."""

    output_file.parent.mkdir(exist_ok=True, parents=True)

    files = list(sorted(input_dir.rglob("*.dat"), key=lambda p: int(p.stem)))

    # Multiple processes
    # h5file = h5py.File(output_path, "w")
    frames = process_map(
        process1M if data_type == "1M" else process2M,
        zip(
            range(len(files)),
            [files] * len(files),
            [max_time] * len(files),
        ),
        total=len(files),
        #max_workers=16,
    )
    # print(type(frames), len(frames))

    # # Single process
    # frames = []
    # for i in range(len(files)):
    #     if data_type == "1M":
    #         frames.append(process1M((i, files, max_time)))
    #     else:
    #         frames.append(process2M((i, files, max_time)))
    # frames = np.concatenate(frames, axis=0)


    # print(f"Total frames: {frames.shape[0]}")
    data = prepareData(frames, min_frame, max_frame)

    # h5file.create_dataset("packedbits", data=frames)
    # print("Begin to compress")
    # data_compressed = np.packbits(data,axis=2)
    print("Begin to save")
    # default little order
    np.savez_compressed(output_file, data)
    print("End save")


def read_all(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_path in input_dir.iterdir():
        if data_path.is_dir():
            read_dat(
                data_path,
                output_dir.joinpath(data_path.name + ".npz"),
                min_frame=0,
                max_frame=2000,
            )
            print(f"Finish {data_path.name}")


def main(input_dir: Path, output_dir: Path):
    # tyro.cli(read_dat)
    read_all(input_dir=input_dir, output_dir=output_dir)
    # Path("/mnt/sdb/data/spike/2030proj/raw"),
    # Path("/mnt/sdb/data/spike/2030proj/planes"),


def mainForSpiketest():
    input_dir = Path("/media/knocking/Seagate_Basic/temdata/")
    output_dir = Path("/media/knocking/Seagate_Basic/temdata_out/")
    cameras = list(input_dir.iterdir())
    for camera in cameras:
        if camera.name != "2M":
            lights = list(camera.iterdir())
            for lux in tqdm(lights):
                output_dir.mkdir(parents=True, exist_ok=True)
                read_dat(
                    lux,
                    output_dir / camera.name / (lux.name + ".npz"),
                    #min_frame=1000,
                    #max_frame=3000,
                    data_type="1M",
                )
        else:
            lights = list(camera.iterdir())
            lights = sorted(lights, key=lambda x: float(x.stem))
            for lux in tqdm(lights):
                output_dir.mkdir(parents=True, exist_ok=True)
                read_dat(
                    lux,
                    output_dir / camera.name / (lux.name + ".npz"),
                    #min_frame=,
                    #max_frame=3000,
                    data_type="2M",
                )
    # Path("/mnt/sdb/data/spike/2030proj/raw"),
    # Path("/mnt/sdb/data/spike/2030proj/planes"),


if __name__ == "__main__":
    # tyro.cli(main)
    mainForSpiketest()
