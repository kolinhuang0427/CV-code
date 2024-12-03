import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import functions as fun
import helper

# 1. Load the two temple images
im1 = io.imread("i14.jpg")
im2 = io.imread("i15.jpg")

# Load predefined corresponding points in the 2 images
with np.load("some_corresp.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']

# 2. Run eight_point to compute F
F = fun.eight_point(pts1, pts2, np.max(pts1.shape))
print("Fundamental matrix\n", F, end="\n\n")
# for visualization
# helper.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/pointsForReconstruction.npz
with np.load("pointsForReconstruction.npz") as data:
    pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
pts2 = fun.epipolar_correspondences(im1, im2, F, pts1)
# for visualization, calls epipolar_correspondences inside
# helper.epipolarMatchGUI(im1, im2, F)

# 5. Load camera intrinsics data from data/intrinsics.npz, assuming we have the matrices
with np.load("intrinsics.npz") as data:
    K1 = K2 = data["K"]

# 6. Compute the essential matrix
E = fun.essential_matrix(F, K1, K2)
print("Essential Matrix\n", E, "\n")

# 7. Create the camera projection matrix P1
# Assume that extrinsics for camera 1 = (I | 0)
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

# 8. Use camera2 to get 4 camera projection matrices M2
M2s = helper.camera2(E)

# 9. Run triangulate using the projection matrices and find out correct M2, P2, and pts_3d
pts_3d = None
M2 = None
P2 = None
best = 0
count = 0
for i in range(4):
    P2_candidate = K2 @ M2s[:, :, i]
    
    pts_3d_candidate = fun.triangulate(P1, pts1, P2_candidate, pts2)
    count = np.count_nonzero(pts_3d_candidate[:, 2] > 0)
    if count > best: 
        best = count
        pts_3d = pts_3d_candidate
        P2 = P2_candidate
        M2 = M2s[:, :, i]
        break

# 10. (for testing) Get reprojection error 
pts_3d_homogeneous = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

# Project 3D points back to 2D using the camera projection matrix
projected_pts_2d_homogeneous = P2 @ pts_3d_homogeneous.T  # (3xN)

projected_pts_2d = projected_pts_2d_homogeneous[:2, :] / projected_pts_2d_homogeneous[2, :]  # (2xN)

# Compute the Euclidean distance between the projected points and the original points
error = np.linalg.norm(pts1.T - projected_pts_2d, axis=0)
reprojectionError = np.mean(error)
print("\nReprojection error: ", reprojectionError)

# 11. (to view) Scatter plot the correct 3D points
# From https://pythonprogramming.net/matplotlib-3d-scatterplot-tutorial/
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xData = pts_3d[:, 0]
yData = pts_3d[:, 1]
zData = pts_3d[:, 2]

ax.scatter(xData, yData, zData, c='b', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout(pad=0.1)
plt.show()

# 12. Save the computed extrinsic parameters (R1,R2,t1,t2) to results/extrinsics.npz
R1 = np.eye(3)
t1 = np.zeros((3,1))
R2 = M2[:, :3]
t2 = M2[:, 3:]
np.savez('extrinsics.npz', R1=R1, t1=t1, R2=R2, t2=t2)