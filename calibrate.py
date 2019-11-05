#encoding=utf-8
import numpy as np
import cv2
import os
import time

# 定义全局变量
map_x, map_y = None, None

# 消除畸变
def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print('roi ', roi)
    # 耗时操作
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # 替代方案(节省时间)/map_x, map_y使用全局变量更加节省时间
    global map_x, map_y
    if map_x is None and map_y is None:
        # 计算一个从畸变图像到非畸变图像的映射(只需要执行一次, 找出映射关系即可)
        map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # 使用映射关系对图像进行去畸变
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    # 裁剪图片
    # x, y, w, h = roi
    # if roi != (0, 0, 0, 0):
    #     dst = dst[y:y + h, x:x + w]
    return dst


# 标注图片(生成相机内外参数)
def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 横向角点(不算最外边缘)
    Nx_cor = 9
    # 竖向角点
    Ny_cor = 6

    objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    path = 'D:/Test/Python/CameraCalibrate/pic'
    pic_list = os.listdir(path)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(pic_list)):
        pic_path = os.path.join(path, pic_list[i])
        if os.path.isfile(pic_path):
            frame = cv2.imread(pic_path)
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 查找棋盘格角点信息
            ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
            # If found, add object points, image points
            # 精细化角点信息
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 绘制查找到的角点
            cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ret)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    global mtx, dist
    # 标定, mtx 是相机内参, dist 是畸变, rvecs, tvecs分别是旋转矩阵和平移矩阵代表外参
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)

    mean_error = 0
    for i in range(len(objpoints)):
        # 将角点的物理坐标, 标定得到的外参重新投影得到新的角点的图像坐标
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # 将新的图像坐标与之前检测角点时的真实图像坐标对比, 以此来衡量标定的精确性
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))
        # # When everything done, release the capture
    # 将标定结果序列化, 保存到本地
    np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])


if __name__ == '__main__':
    cap = cv2.VideoCapture('D:\Test\Python\CameraCalibrate\output_calibrate.mp4')
    mtx = []
    dist = []
    try:
        npzfile = np.load('calibrate.npz')
        mtx = npzfile['mtx']
        dist = npzfile['dist']
    except IOError:
        calibrate()

    print('dist', dist[0:4])
    frame_id = 0

    start = time.time()

    while (True):
        ret, frame = cap.read()
        if ret:
            frame = undistortion(frame, mtx, dist[0:4])
            # Display the resulting frame
            # cv2.imshow('frame', frame)
            # cv2.imwrite('picture/'+str(frame_id)+'.jpg', frame)
            # cv2.waitKey(5)
            frame_id += 1
        else:
            end = time.time()
            break
    print(end - start)
    cap.release()
    cv2.destroyAllWindows()