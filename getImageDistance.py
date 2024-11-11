import numpy as np
from utils import chi2dist

def get_image_distance(hist1, hist2, method):

    # -----fill in your implementation here --------
    if method == 'euclidean':
        # Euclidean distance
        dist = np.linalg.norm(hist1 - hist2)
    elif method == 'chi2':
        # Chi-squared distance using chi2dist function
        dist = chi2dist(hist1, hist2)
    else:
        raise ValueError("Method must be 'euclidean' or 'chi2'")
    
    return dist

    # ----------------------------------------------
