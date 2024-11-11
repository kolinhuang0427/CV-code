import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h, _ = np.histogram(wordMap, bins=np.arange(dictionarySize + 1), density=False)
    
    # L1 Normalize the histogram
    h = h / np.sum(h)

    # ----------------------------------------------
    
    return h
