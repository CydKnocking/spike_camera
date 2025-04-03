import cv2
import numpy as np
import glob
 
# 相机标定
#设置棋盘格w和h方向的角点数量
w_corners = 11
h_corners = 8
#设置图像路径
images=glob.glob(r"C:\\img_path\\*.bmp")  #黑白棋盘的图片路径
 
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
 
# 获取标定板角点的位置
objp = np.zeros((w_corners * h_corners , 3), np.float32)
objp[:, :2] = np.mgrid[0:w_corners , 0:h_corners ].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
 
obj_points = []  # 存储3D点
img_points = []  # 存储2D点

i=0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w_corners , h_corners ), None)
 
    if ret:
 
        obj_points.append(objp)
 
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
 
        cv2.drawChessboardCorners(img, (w_corners , h_corners ), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i+=1
        
        #显示标定过程
        new_size = (800,600)
        resized_img = cv2.resize(img, new_size)
        cv2.imshow('img', resized_img)
        cv2.waitKey(150)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵--内参
print("dist:\n", dist)  # 畸变系数--内参 
print("rvecs:\n", rvecs)  # 旋转向量--外参
print("tvecs:\n", tvecs ) # 平移向量--外参
