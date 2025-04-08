import os
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from spike import TFI, TFP
import cv2
from glob import glob
import matplotlib.pyplot as plt

def read_npy():
    data = np.load("/media/knocking/Seagate_Basic/temdata/10w/planes/20241214165516744.npy")
    print(data.shape, data.dtype, data.max(), data.min())
    print(data[0, :5, :])

def test_different_alpha_beta_chessboard(plane_file, save_path):
    # 用TFI算法测试不同的alpha和beta

    data = np.load(plane_file)
    data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    data = data[:1000]

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

def reconstruct(plane_file, save_path):
    data = np.load(plane_file)
    data = np.unpackbits(data, axis=2, bitorder="little").astype(np.uint8)
    data = data[:1000]
    # print(data.shape, data.dtype, data.max(), data.min()) # (60000, 250, 400) uint8 1 0
    # print(data[0, :5, :])

    # Set beta to increase the brightness of the image
    alpha = 15.0
    beta = 40

    # TFI
    tfi = TFI()
    frames = tfi.reconstruct(data)
    # print(frames.shape, frames.dtype, frames.max(), frames.min()) # (6000, 250, 400) float32 1.0 0.0015600624
    # print(frames[0, :5, :])
    frames = (frames * 255).astype(np.uint8)
    for idx, frame in enumerate(tqdm(frames, desc="saving TFI", leave=False)):
        # flip the frame horizontally
        frame = cv2.flip(frame, 1)
        # use cv2 to show the frame in float32 and gray scale
        _f = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        # put a text on the frame
        # cv2.putText(_f, f"{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow("frame_TFI", _f)
        # cv2.imshow("frame_TFI", frame)
        cv2.imwrite(str(save_path / f"{idx}.png"), _f)
        # cv2.waitKey(1)
    # cv2.destroyAllWindows()

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

def convert_planes_to_png():
    planes_path = "/media/knocking/Seagate_Basic/calib_spike_0405_1"
    planes = sorted(glob(os.path.join(planes_path, "planes/*.npy")))
    save_path = Path(planes_path) / "reconstruct_TFI"
    save_path.mkdir(exist_ok=True)
    for plane in tqdm(planes, desc="reconstructing"):
        # print(plane)
        file_name = Path(plane).name.split(".")[0]
        save_p = save_path / f"{file_name}"
        save_p.mkdir(exist_ok=True)
        reconstruct(plane, save_p)

if __name__ == '__main__':
    convert_planes_to_png()
    # test_different_alpha_beta()
