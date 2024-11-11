"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import scipy
from numpy.linalg import svd
import helper
import cv2
from scipy.signal import correlate2d

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # Normalize points
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts1_norm = pts1_h @ T
    pts2_norm = pts2_h @ T

    # Create matrix A for the 8-point algorithm
    A = np.array([pts1_norm[:, 0] * pts2_norm[:, 0], 
                  pts1_norm[:, 0] * pts2_norm[:, 1],
                  pts1_norm[:, 0],
                  pts1_norm[:, 1] * pts2_norm[:, 0],
                  pts1_norm[:, 1] * pts2_norm[:, 1],
                  pts1_norm[:, 1],
                  pts2_norm[:, 0],
                  pts2_norm[:, 1],
                  np.ones(pts1.shape[0])]).T
    
    # Solve Af = 0
    _, _, V = svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, V = svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    # Unscale F
    F = T.T @ F @ T

    return helper.refineF(F, pts1, pts2)

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1):
    pts2 = np.zeros_like(pts1)
    half_w = 105 // 2
    search_range = 20
    
    for i, pt1 in enumerate(pts1):
        x1 = np.array([pt1[0], pt1[1], 1])
        epipolar_line = F @ x1

        best_match = None
        best_score = -1
        
        # Restrict y_vals to a search window around pt1
        y_vals = np.arange(max(0, pt1[1] - search_range), min(im2.shape[0], pt1[1] + search_range))
        x_vals = (epipolar_line[2] -epipolar_line[1] * y_vals) / epipolar_line[0] 

        if pt1[0] - half_w >= 0 and pt1[0] + half_w < im1.shape[1] and pt1[1] - half_w >= 0 and pt1[1] + half_w < im1.shape[0]:
            patch1 = im1[pt1[1]-half_w:pt1[1]+half_w+1, pt1[0]-half_w:pt1[0]+half_w+1]

        for y2, x2 in zip(y_vals.astype(int), x_vals.astype(int)):
            if x2 < half_w or x2 >= im2.shape[1] - half_w or y2 < half_w or y2 >= im2.shape[0] - half_w:
                #print(f"Skipping point {pt1} due to 1111.")
                continue
            
            patch2 = im2[y2-half_w:y2+half_w+1, x2-half_w:x2+half_w+1]

            ncc = np.sum(patch1 * patch2) / (np.sqrt(np.sum(patch1**2)) * np.sqrt(np.sum(patch2**2)))
            
            if ncc > best_score:
                best_match = [x2, y2]
                best_score = ncc

        if best_match is not None:
            pts2[i] = best_match
        else:
            pts2[i] = pts1[i]
            #print(f"No match found for point {pt1}. Defaulting to pts1[i].")

    return pts2

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    pts3d = []
    
    for i in range(pts1.shape[0]):
        #print(f"Processing point pair {i}: pts1={pts1[i]}, pts2={pts2[i]}")
        y = pts1[i, 1]
        x = pts1[i, 0]
        y1 = pts2[i, 1]
        x1 = pts2[i, 0]
        A = np.array([
            y * P1[2, :] - P1[1, :],
            P1[0, :] - x * P1[2, :],
            y1 * P2[2, :] - P2[1, :],
            P2[0, :] - x1 * P2[2, :]
        ])
        #print("Matrix A", A)
        _, _, V = svd(A)
        X = V[-1]
        X = X / X[3]
        #print(f"Triangulated 3D point: {X[:3]}")
        pts3d.append(X[:3])

    return np.array(pts3d)


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Step 1: Compute the optical centers c1 and c2
    c1 = -R1.T @ t1  # Optical center of camera 1
    c2 = -R2.T @ t2  # Optical center of camera 2

    # Step 2: Compute the rectification rotation matrix Re
    T = c2 - c1  # Translation vector from camera 1 to camera 2
    r1 = T / np.linalg.norm(T)  # New x-axis parallel to the baseline
    r1 = r1.flatten()
    Tx, Ty, Tz = T.flatten()
    r2 = np.array([-Ty, Tx, 0])
    r2 = r2 / np.linalg.norm(r2)  # Normalize to a unit vector
    r3 = np.cross(r1, r2)
    r3 = r3 / np.linalg.norm(r3)  # Normalize to a unit vector
    Re = np.column_stack((r1, r2, r3)) 

    # Step 3: Compute the new rotation matrices
    R1p = R1 @ Re.T  # Use transpose of Re for correct transformation
    R2p = R2 @ Re.T # Use transpose of Re for correct transformation

    # Step 4: Compute the new intrinsic parameters
    K1p = K2.copy()  # New intrinsic parameters
    K2p = K2.copy()  # Same as K2 in the rectified view

    # Step 5: Compute the new translation vectors
    t1p = -R1p @ c1  # New translation vector for camera 1
    t2p = -R2p @ c2  # New translation vector for camera 2

    # Step 6: Compute the rectification matrices
    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)  # Rectification matrix for camera 1
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)  # Rectification matrix for camera 2

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    H1, W1 = im1.shape
    dispM = np.zeros((H1, W1))
    
    half_w = (win_size - 1) // 2
    for y in range(half_w, H1 - half_w):
        for x in range(half_w, W1 - half_w):
            best_disp = 0
            best_score = float('inf')
            
            # Extract the patch from the first image
            patch1 = im1[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1]
            
            for d in range(min(max_disp, x - half_w)):
                # Check for valid x position in the second image
                if x - d - half_w < 0 or x - d + half_w >= W1:
                    continue
                
                # Extract the patch from the second image
                patch2 = im2[y - half_w:y + half_w + 1, (x - d) - half_w:(x - d) + half_w + 1]
                
                # Compute the SSD
                ssd = np.sum((patch1 - patch2) ** 2)
                
                if ssd < best_score:
                    best_score = ssd
                    best_disp = d
            
            dispM[y, x] = best_disp

    return dispM



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # Compute the baseline
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    baseline = np.linalg.norm(c2 - c1)
    
    # Assume focal length is the first element of K1
    f = K1[0, 0]
    
    # Create the depth map
    depthM = np.zeros_like(dispM)
    with np.errstate(divide='ignore', invalid='ignore'):
        depthM = baseline * f / dispM
    
    # Set depth to 0 wherever dispM is 0 to avoid division by zero
    depthM[dispM == 0] = 0
    
    return depthM



"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
