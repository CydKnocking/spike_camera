raise DeprecationWarning("This file is deprecated, please use calib_stereo.py instead.")

import cv2
import numpy as np
import glob
from natsort import natsorted

class Calibrator(object):
    """
    相机标定，同时可对realsense和spike相机进行标定
    """

    def __init__(self, img_paths, w_c=11, h_c=8, crop_factor=None):
        self.w_corners = w_c
        self.h_corners = h_c
        self.img_paths = img_paths
        self.crop_factor = crop_factor
        self._load_imgs()

    def _load_imgs(self):
        self.images = []
        for _p in self.img_paths:
            # 读取图片以灰度图的方式
            img = cv2.imread(_p, cv2.IMREAD_GRAYSCALE)
            if self.crop_factor is not None:
                # 裁剪图片
                h, w = img.shape
                img = img[int(h * self.crop_factor):int(h * (1 - self.crop_factor)), int(w * self.crop_factor):int(w * (1 - self.crop_factor))]
            self.images.append(img)
        self.size = self.images[0].shape[::-1]
    
    def calibrate(self):
        # 获取标定板角点的位置
        objp = np.zeros((self.w_corners * self.h_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w_corners, 0:self.h_corners].T.reshape(-1, 2)
        
        obj_points = []  # 存储3D点
        img_points = []  # 存储2D点

        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

        for img, i_p in zip(self.images, self.img_paths):
            ret, corners = cv2.findChessboardCorners(img, (self.w_corners, self.h_corners), None)

            if not ret:
                print(f"Can't find chessboard corners in {i_p}")
                continue

            print(f"Processing {i_p}")
            
            obj_points.append(objp)
            
            corners2 = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            
            cv2.drawChessboardCorners(img, (self.w_corners, self.h_corners), corners, ret)
            # 显示标定过程
            new_size = (800, 600)
            resized_img = cv2.resize(img, new_size)
            cv2.imshow('img', resized_img)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.size, None, None)

        return ret, mtx, dist, rvecs, tvecs

def calib_realsense_cam(img_path, w_c=11, h_c=8):
    # # 设置棋盘格w和h方向的角点数量
    # w_corners = 11
    # h_corners = 8

    # 查看图片路径下有多少文件夹
    img_folders = glob.glob(img_path + "/*")
    img_folders = natsorted(img_folders)
    # print(img_folders)

    # 取每个文件夹下标号最后的图片
    images = []
    for folder in img_folders:
        img = glob.glob(folder + "/*.png")
        img = natsorted(img)
        images.append(img[-1])
    # print(images)

    # 创建Calibrator对象
    calibrator = Calibrator(images, w_c, h_c, crop_factor=0.25)
    ret, mtx, dist, rvecs, tvecs = calibrator.calibrate()

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵--内参
    print("dist:\n", dist)  # 畸变系数--内参
    print("rvecs:\n", rvecs)  # 旋转向量--外参
    print("tvecs:\n", tvecs)  # 平移向量--外参

def calib_spike_cam(img_path, w_c=11, h_c=8):
    # # 设置棋盘格w和h方向的角点数量
    # w_corners = w_c
    # h_corners = h_c

    # 查看图片路径下有多少文件夹
    img_folders = glob.glob(img_path + "/*")
    img_folders = natsorted(img_folders)
    # print(img_folders)

    # 取每个文件夹下标号正中间的图片
    images = []
    for folder in img_folders:
        img = glob.glob(folder + "/*.png")
        img = natsorted(img)
        images.append(img[len(img) // 2])
    # print(images)

    # 创建Calibrator对象
    calibrator = Calibrator(images, w_c, h_c)
    ret, mtx, dist, rvecs, tvecs = calibrator.calibrate()

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵--内参
    print("dist:\n", dist)  # 畸变系数--内参
    print("rvecs:\n", rvecs)  # 旋转向量--外参
    print("tvecs:\n", tvecs)  # 平移向量--外参

def debug_test_spike_cam(img_path, w_c=11, h_c=8):
    # # 查看图片路径下有多少文件夹
    # img_folders = glob.glob(img_path + "/*")
    # img_folders = natsorted(img_folders)
    # # print(img_folders)

    # # 取每个文件夹下标号最后的图片
    # images = []
    # for folder in img_folders:
    #     img = glob.glob(folder + "/*.png")
    #     img = natsorted(img)
    #     images.append(img[-1])
    # # print(images)

    images = glob.glob(img_path + "/*.png")
    images = natsorted(images)

    # 创建Calibrator对象
    calibrator = Calibrator(images, w_c, h_c)
    ret, mtx, dist, rvecs, tvecs = calibrator.calibrate()

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵--内参
    print("dist:\n", dist)  # 畸变系数--内参
    print("rvecs:\n", rvecs)  # 旋转向量--外参
    print("tvecs:\n", tvecs)  # 平移向量--外参

if __name__ == '__main__':
    calib_spike_cam(img_path="/media/knocking/Seagate_Basic/calib_spike_realsense/4/spike")
    calib_realsense_cam(img_path="/media/knocking/Seagate_Basic/calib_spike_realsense/4/realsense")
    # debug_test_spike_cam(img_path="/media/knocking/Seagate_Basic/calib_spike/debug_test_different_alpha_beta")

