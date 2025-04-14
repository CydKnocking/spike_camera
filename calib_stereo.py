import cv2
import numpy as np
import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class Calibrator(object):
    """
    相机标定，同时可对realsense和spike相机进行标定
    """

    def __init__(self, img_paths, w_c=11, h_c=8):
        self.w_corners = w_c
        self.h_corners = h_c
        self.img_paths = img_paths
        self._load_imgs()

    '''
    def _load_imgs(self):
        self.images = []
        for _p in self.img_paths:
            # 读取图片以灰度图的方式
            img = cv2.imread(_p, cv2.IMREAD_GRAYSCALE)
            self.images.append(img)
        self.size = self.images[0].shape[::-1]
    '''
    # 将读取的图像缩放至(640, 480)
    def _load_imgs(self):
        self.images = []
        target_size = (640, 480)  # (width, height)
    
        for _p in self.img_paths:
            # 读取图片为灰度图
            img = cv2.imread(_p, cv2.IMREAD_GRAYSCALE)
        
            # 检查尺寸是否一致，不一致则缩放
            if img.shape[::-1] != target_size:  # img.shape 是 (height, width)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  
            self.images.append(img)

        self.size = target_size
    
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

        return ret, mtx, dist, rvecs, tvecs, obj_points, img_points

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
    calibrator = Calibrator(images, w_c, h_c)
    ret, mtx, dist, rvecs, tvecs, obj_points, img_points = calibrator.calibrate()

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵--内参
    print("dist:\n", dist)  # 畸变系数--内参
    print("rvecs:\n", rvecs)  # 旋转向量--外参
    print("tvecs:\n", tvecs)  # 平移向量--外参
    
    return mtx, dist, rvecs, tvecs, obj_points, img_points

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
    ret, mtx, dist, rvecs, tvecs, obj_points, img_points = calibrator.calibrate()

    print("ret:", ret)
    print("mtx:\n", mtx)  # 内参数矩阵--内参
    print("dist:\n", dist)  # 畸变系数--内参
    print("rvecs:\n", rvecs)  # 旋转向量--外参
    print("tvecs:\n", tvecs)  # 平移向量--外参
    
    return mtx, dist, rvecs, tvecs, obj_points, img_points

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


# 计算平均重投影误差
def compute_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0

    for i in range(len(obj_points)):
        # 使用相机内参将3D点投影到2D图像平面
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                               camera_matrix, dist_coeffs)
        img_points_proj = img_points_proj.reshape(-1, 2)
        img_points_gt = img_points[i].reshape(-1, 2)

        # 计算当前图像的误差（欧几里得距离）
        error = cv2.norm(img_points_gt, img_points_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(obj_points[i])

    # 平均重投影误差（每个点的平均误差）
    mean_error = np.sqrt(total_error / total_points)
    return mean_error

def calib_stereo(mtx_A, dist_A, rvecs_A, tvecs_A, obj_points_A, img_points_A, 
                   mtx_B, dist_B, rvecs_B, tvecs_B, obj_points_B, img_points_B):
    
    target_size = (640, 480)
    # 计算相对位姿
    R_rel_list, t_rel_list = [], []
    for rvec_A, tvec_A, rvec_B, tvec_B in zip(rvecs_A, rvecs_B, tvecs_A, tvecs_B):
        R_A, _ = cv2.Rodrigues(rvec_A.astype(np.float64))
        R_B, _ = cv2.Rodrigues(rvec_B.astype(np.float64))
    
        R_rel = R_B @ R_A.T
        t_rel = tvec_B.reshape(3,1) - R_rel @ tvec_A.reshape(3,1)
    
        R_rel_list.append(R_rel)
        t_rel_list.append(t_rel.flatten())


    # 平均旋转矩阵（使用四元数平均）
    rotations = [R.from_matrix(R_rel) for R_rel in R_rel_list]
    
    # 将旋转转换为四元数并计算均值
    
    #mean_rot = R.mean(rotations) #有问题，改成手动算
    
    #手动计算
    quats = np.array([r.as_quat() for r in rotations])
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)  # 单位化
    mean_rot = R.from_quat(mean_quat)
    
    R_rel_avg = mean_rot.as_matrix()

    # 平均平移向量
    t_rel_avg = np.mean(t_rel_list, axis=0)

    # 立体校正使用统一尺寸
    rectify_scale = 0  # 0裁剪黑边，1保留所有像素
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_A, dist_A,         
        mtx_B, dist_B,         
        target_size,           
        R_rel_avg, t_rel_avg,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=rectify_scale
    )
    
    # 重投影误差
    error_A = compute_reprojection_error(obj_points_A,img_points_A,rvecs_A,tvecs_A,mtx_A,dist_A)
    error_B = compute_reprojection_error(obj_points_B,img_points_B,rvecs_B,tvecs_B,mtx_B,dist_B)
    
    # 打印关键参数
    print("\n=========== 标定结果 ===========")
    print(f"1. 相机间旋转矩阵 R:\n{R_rel_avg}")
    print(f"\n2. 相机间平移向量 T (mm):\n{t_rel_avg}")
    print(f"\n3. 校正投影矩阵 P1:\n{P1}")
    print(f"\n4. 视差转深度矩阵 Q:\n{Q}")
    print(f"\n5. 重投影误差: \n spike_cam: {np.mean(error_A):.4f} pixels \n realsense: {np.mean(error_B):.4f} pixels")



if __name__ == '__main__':
    mtx_A, dist_A, rvecs_A, tvecs_A, obj_points_A, img_points_A = calib_spike_cam(img_path="./data/spike")
    mtx_B, dist_B, rvecs_B, tvecs_B, obj_points_B, img_points_B = calib_realsense_cam(img_path="./data/realsense")
    calib_stereo(mtx_A, dist_A, rvecs_A, tvecs_A, obj_points_A, img_points_A, 
                   mtx_B, dist_B, rvecs_B, tvecs_B, obj_points_B, img_points_B)
    
    #calib_spike_cam(img_path="/media/knocking/Seagate_Basic/calib_spike_realsense/1/spike")
    #calib_realsense_cam(img_path="/media/knocking/Seagate_Basic/calib_spike_realsense/1/realsense")
    #debug_test_spike_cam(img_path="/media/knocking/Seagate_Basic/calib_spike/debug_test_different_alpha_beta")
    
