import numpy as np


def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    H, W = I.shape[:2]
    
    y_coords = np.random.randint(0, H, alpha)
    x_coords = np.random.randint(0, W, alpha)
    
    points = np.column_stack((y_coords, x_coords))

    # ----------------------------------------------
    return points
