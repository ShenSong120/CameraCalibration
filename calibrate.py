#encoding=utf-8
import numpy as np
import cv2
import os


def undistortion(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('roi ', roi)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        dst = dst[y:y + h, x:x + w]
    return dst

def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    Nx_cor = 9
    Ny_cor = 6

    objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    path = r'E:\software\DaHengVision\install_file\GalaxySDK\pic'
    pic_list = os.listdir(path)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(pic_list)):
        pic_path = os.path.join(path, pic_list[i])
        if os.path.isfile(pic_path):
            frame = cv2.imread(pic_path)
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
            # If found, add object points, image points
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ret)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    global mtx, dist

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx, dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))
        # # When everything done, release the capture
    np.savez('calibrate.npz', mtx=mtx, dist=dist[0:4])


if __name__ == '__main__':
    cap = cv2.VideoCapture('D:\Test\Python\CameraCalibrate\output.mp4')
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
    while (True):
        ret, frame = cap.read()
        if ret:
            frame = undistortion(frame, mtx, dist[0:4])
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imwrite('picture/'+str(frame_id)+'.jpg', frame)
            cv2.waitKey(5)
            frame_id += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()