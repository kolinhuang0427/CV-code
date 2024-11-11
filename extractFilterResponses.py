import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import *
import matplotlib.pyplot as plt

def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    lab_image = rgb2lab(I)
    
    H, W, _ = lab_image.shape
    filterResponses = np.zeros((H, W, len(filterBank) * 3), dtype=np.float32)
    
    # Apply each filter to each channel in the Lab color space
    response_index = 0
    for filter in filterBank:
        for channel in range(3):  # Apply to L, a, and b channels
            # Filter the current channel with the current filter
            filtered_image = imfilter(lab_image[:, :, channel], filter)
        
            filterResponses[:, :, response_index] = filtered_image
            response_index += 1

    # ----------------------------------------------
    
    return filterResponses

# image = cv.imread('../data/desert/sun_aafqfjpechscyidz.jpg')

# filterBank = create_filterbank()

# filter_responses = extract_filter_responses(image, filterBank)

# filter_indices = [0, 20, 40]
# responses_to_display = [filter_responses[:, :, i] for i in filter_indices]

# fig, axes = plt.subplots(1, 4, figsize=(15, 5))
# axes[0].imshow(image)
# axes[0].set_title("Original Image")
# axes[0].axis("off")

# for i, response in enumerate(responses_to_display):
#     axes[i + 1].imshow(response, cmap='gray')
#     axes[i + 1].set_title(f"Filter Response {filter_indices[i]}")
#     axes[i + 1].axis("off")

# plt.show()