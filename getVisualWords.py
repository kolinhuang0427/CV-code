import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    filterResponses = extract_filter_responses(I, filterBank)
    
    # Reshape filterResponses to (H*W, num_filters)
    H, W, num_filters = filterResponses.shape
    reshaped_responses = filterResponses.reshape(H * W, num_filters)
    
    # Compute the distance from each pixel response to each dictionary word
    distances = cdist(reshaped_responses, dictionary, metric='euclidean')
    
    # Find the index of the closest dictionary word for each pixel response
    wordMap = np.argmin(distances, axis=1).reshape(H, W)

    # ----------------------------------------------

    return wordMap

