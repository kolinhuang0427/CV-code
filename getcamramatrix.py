import cv2
import numpy as np
import os

# Prepare object points for the checkerboard (e.g., 6x9 corners)
checkerboard_size = (6, 8)
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

for fname in os.listdir():
    if fname.endswith('i15.jpg'):
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
# img = cv2.imread('personleft.jpg')
# print(img.shape)
# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(len(objpoints))
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)
print(rvecs)
print(tvecs)

np.savez('i15.npz', ret=ret, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)