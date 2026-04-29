from fileinput import filename
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

def read_npy():
    data = np.load("/media/knocking/Seagate_Basic/temdata/10w/planes/20241214165516744.npy")
    print(data.shape, data.dtype, data.max(), data.min())
    print(data[0, :5, :])

def test_different_alpha_beta_chessboard(plane_file, save_path):
    # 用TFI算法测试不同的alpha和beta

    data = np.load(plane_file)
    data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    # data = data[:1000]

    tfi = TFI(interval=20)
    frames = tfi.reconstruct(data)
    frames = (frames * 255).astype(np.uint8)
    frame_example = frames[25]
    success = []
    for beta in range(0, 101, 20):
        for alpha in range(1, 50, 2):
            _f = cv2.convertScaleAbs(frame_example, alpha=alpha, beta=beta)
            # _f = cv2.fastNlMeansDenoising(_f, None, 10, 7, 21)
            # cv2.putText(_f, f"alpha: {alpha}, beta: {beta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imwrite(str(save_path / f"alpha_{alpha}_beta_{beta}.png"), _f)
            # cv2.imshow("frame_example", _f)
            # cv2.waitKey(1000)
            ret, corners = cv2.findChessboardCorners(_f, (11, 8), None)
            if ret:
                print(f"alpha: {alpha}, beta: {beta} success")
                cv2.drawChessboardCorners(_f, (11, 8), corners, ret)
                cv2.putText(_f, f"alpha: {alpha}, beta: {beta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("frame_example", _f)
                cv2.waitKey(0)
                success.append([alpha, beta])
            else:
                print(f"alpha: {alpha}, beta: {beta} failed", end="\r")
    cv2.destroyAllWindows()
    success = np.array(success)
    plt.scatter(success[:, 0], success[:, 1])
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("alpha vs beta")
    plt.show()

def reconstruct_stream(plane_file):
    start_time = time()
    print(f"reconstructing {plane_file}")
    data = np.load(plane_file)
    print(f"data loaded, shape: {data.shape}, time: {time() - start_time}")
    start_time = time()
    data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    print(f"data unpacked, shape: {data.shape}, time: {time() - start_time}")

    # TFI
    tfi = TFI(device="cuda")
    for idx, frame in enumerate(tfi.reconstruct_stream(data)):
        print(f"frame {idx}, shape: {frame.shape}")
        cv2.imshow(f"frame", frame)
        cv2.waitKey(1)

def reconstruct(plane_file, save_path, save_h5_name=None):
    print(f"reconstructing {plane_file}")
    data = np.load(plane_file)
    print(f"data loaded, shape: {data.shape}")
    data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    print(f"data unpacked, shape: {data.shape}")
    # data = data[:1000]
    # print(data.shape, data.dtype, data.max(), data.min()) # (60000, 250, 400) uint8 1 0
    # print(data[0, :5, :])

    # Set beta to increase the brightness of the image
    alpha = 10.0
    beta = 20

    # TFI
    tfi = TFI(device="cuda")
    frames = tfi.reconstruct(data)
    print(f"frames reconstructed, shape: {frames.shape}")
    # print(frames.shape, frames.dtype, frames.max(), frames.min()) # (6000, 250, 400) float32 1.0 0.0015600624
    # print(frames[0, :5, :])
    frames = (frames * 255).astype(np.uint8)

    ### Save frames to h5 file
    if save_h5_name is not None:
        print(f"saving frames to h5 file: {save_path / save_h5_name}")
        with h5py.File(save_path / save_h5_name, "w") as f:
            f.create_dataset("frames", data=frames[:, :, ::-1])  # Remember to flip the frame horizontally

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

    # print(f"min_file_idx: {min_file_idx}, max_file_idx: {max_file_idx}")

    # # test different setting of alpha and beta
    # frame_example = frames[3440]
    # for beta in range(0, 101, 20):
    #     for alpha in range(1, 21, 2):
    #         _f = cv2.convertScaleAbs(frame_example, alpha=alpha, beta=beta)
    #         cv2.putText(_f, f"alpha: {alpha}, beta: {beta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #         cv2.imshow("frame_example", _f)
    #         cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # assert 1==0

    # # TFP
    # tfp = TFP()
    # frames = tfp.reconstruct(data)
    # # print(frames.shape, frames.dtype, frames.max(), frames.min()) # (6000, 250, 400) float32 1.0 0.0
    # # print(frames[0, :5, :])
    # frames = (frames * 255).astype(np.uint8)
    # for idx, frame in enumerate(frames):
    #     _f = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    #     cv2.imwrite(str(save_path / f"{idx}.png"), _f)
    # #     cv2.imshow("frame_TFP", cv2.convertScaleAbs(frame, alpha=alpha, beta=beta))
    # #     # cv2.imshow("frame_TFP", frame)
    # #     cv2.waitKey(1)
    # # cv2.destroyAllWindows()

def test_different_alpha_beta():
    planes_path = "/media/knocking/Seagate_Basic/20250404221957494/planes/20250404221957494.npy"
    save_path = "/media/knocking/Seagate_Basic/20250404221957494/debug_test_different_alpha_beta"
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    test_different_alpha_beta_chessboard(planes_path, save_path)

def test_stream_reconstruct():
    planes_path = "./temdata_250804/planes_splits/20250804152203137_0_50.npy"
    reconstruct_stream(planes_path)
    assert False, "Debug finished"

def convert_planes_to_png(file_prefix=""):
    planes_path = "/media/knocking/Seagate_Basic/calib_spike_250804_seqs"
    planes = sorted(glob(os.path.join(planes_path, f"planes_split/{file_prefix}*.npy")))
    # planes = planes[2:]
    save_path = Path(planes_path) / "reconstruct_TFI"
    save_path.mkdir(exist_ok=True)
    # for plane in tqdm(planes, desc="reconstructing"):
    for _idx, plane in enumerate(planes):
        print("--------------------------------")
        print(plane)
        file_name = Path(plane).name.split(".")[0]
        _fn = file_name.split("_")[0]
        save_p = save_path / f"{_fn}"
        save_p.mkdir(exist_ok=True)

        # start_idx = int(file_name.split("_")[-2])
        # skip_idx = 0 if start_idx <= 0 else 1000
        # skip_end = None
        # print(f"start_idx: {start_idx}, skip_idx: {skip_idx}, skip_end: {skip_end}")
        # if start_idx > 0:
        #     start_idx += 25
        # if _idx < len(planes) - 1:
        #     skip_end = -1000
        # print(f"after adjust start_idx: {start_idx}, skip_idx: {skip_idx}, skip_end: {skip_end}")

        # start_idx = start_idx * 40
        # print(f"after multiply start_idx: {start_idx}, skip_idx: {skip_idx}, skip_end: {skip_end}")
        # continue
        reconstruct(plane, save_p, save_h5_name=f"{file_name}.h5")

if __name__ == '__main__':
    test_stream_reconstruct()
    assert False, "Debug finished"
    file_timestamps = [
        "20250804171943419",
        "20250804171950255",
        "20250804172316892",
        "20250804172733536",
        "20250804173341719",
        "20250804175005846",
        "20250804175243214",
        "20250804175608608",
        "20250804175851208",
    ]
    for file_prefix in file_timestamps:
        convert_planes_to_png(file_prefix)
    # test_different_alpha_beta()
