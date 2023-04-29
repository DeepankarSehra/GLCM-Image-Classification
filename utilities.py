
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

def rescale(feature_arr, img_array):

    entropy_img = entropy(img_array, disk(5))
    entropy_img = (entropy_img - np.min(entropy_img))/(entropy_img.max() - entropy_img.min())

    # Calculate the mean and standard deviation of the grayscale pixels
    mean = np.mean(img_array)
    std_dev = np.std(img_array)

    # Calculate the contrast for each pixel
    contrasts = abs(img_array - mean) / std_dev

    # Normalize the contrast values to a range of 0-255
    contrast = (contrasts / np.max(contrasts)) * 255
    entropy_img = entropy_img*255
    contrast = contrast.reshape(contrast.shape[0], contrast.shape[1], 1)
    entropy_img = entropy_img.reshape(entropy_img.shape[0], entropy_img.shape[1], 1)

    feature = np.concatenate([contrast, entropy_img], axis=-1 )
    feature_arr*=1.0

    featuress = feature.reshape((feature.shape[0]*feature.shape[1], feature.shape[2]))
    return featuress