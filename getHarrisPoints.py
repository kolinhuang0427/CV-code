import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter
import matplotlib.pyplot as plt

def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel x
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel y

    I_x = imfilter(I, sobel_x)  # Gradient x
    I_y = imfilter(I, sobel_y)  # Gradient y

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y

    # Gaussian filter for smoothing
    gaussian_filter = cv.getGaussianKernel(5, 1)
    gaussian_filter = gaussian_filter * gaussian_filter.T

    # Compute the sum of products of derivatives using convolution
    S_xx = imfilter(I_xx, gaussian_filter)
    S_xy = imfilter(I_xy, gaussian_filter)
    S_yy = imfilter(I_yy, gaussian_filter)

    # Compute the Harris response
    det_M = S_xx * S_yy - S_xy * S_xy  # Determinant of M
    trace_M = S_xx + S_yy  # Trace of M
    R = det_M - k * (trace_M ** 2)  # Harris response

    # Find the top α points
    R_flat = R.flatten()
    indices = np.argsort(R_flat)[-alpha:]  # Get indices of top α responses
    points = np.column_stack(np.unravel_index(indices, R.shape))  # Get coordinates

    return points

# image_paths = ['../data/campus/sun_abslhphpiejdjmpz.jpg', '../data/campus/sun_abpxvcuxhqldcvln.jpg', '../data/campus/sun_abwbqwhaqxskvyat.jpg']
# alpha = 100  # Number of corners to detect
# k = 0.04  # Harris detector parameter

# plt.figure(figsize=(12, 8))

# for i, image_path in enumerate(image_paths):
#     # Read image
#     I = cv.imread(image_path)
#     I_rgb = cv.cvtColor(I, cv.COLOR_BGR2RGB)  # Convert to RGB for visualization

#     # Detect corners
#     points = get_harris_points(I_rgb, alpha, k)

#     # Plot image and detected corners
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(I_rgb)
#     plt.scatter(points[:, 1], points[:, 0], c='red', s=10)  # Plot corners in red
#     plt.title(f'Harris Corners - Image {i + 1}')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()