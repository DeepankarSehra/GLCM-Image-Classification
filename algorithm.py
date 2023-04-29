import numpy as np
import skimage
from skimage import io, data
from skimage.transform import resize
import warnings
from PIL import Image
import matplotlib.pyplot as plt
from utilities import rescale
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')


def fastglcm(image, distances, angles, levels, w=5):
    height, width = image.shape                                         # Height and width of the input image
    glcm = np.zeros((levels, levels, len(distances), len(angles)))      # Initialising an array of 0's with the dimensions of 256x256 (8-bit)
    offset = w // 2                                                     # The offset padding for the image
    co_occurs = np.zeros((levels, levels, w, w))                        # The temporary GLCM array for a window initialised with 0's of same dimension


    for i in range(height):
        for j in range(width):
            if i < offset or i >= height - offset or j < offset or j >= width - offset:
                continue
            
            # Update the co-occurrence matrix for the new window column
            for k in range(w):
                y = j - offset + k
                x = i - offset
                i1 = image[x, y]
                for l in range(k+1, w):
                    y1 = j - offset + l
                    i2 = image[x, y1]
                    co_occurs[i1, i2, k, l] += 1
                    co_occurs[i2, i1, l, k] += 1

            # Update the GLCM using the new co-occurrence matrix
            for d, distance in enumerate(distances):
                for a, angle in enumerate(angles):
                    dx = round(distance * np.cos(angle))
                    dy = round(distance * np.sin(angle))
                    x1 = i - offset + dx
                    y1 = j - offset + dy
                    x2 = i - offset - dx
                    y2 = j - offset - dy
                    if x1 < 0 or x1 >= height or y1 < 0 or y1 >= width:
                        continue
                    if x2 < 0 or x2 >= height or y2 < 0 or y2 >= width:
                        continue
                    glcm[:, :, d, a] += co_occurs[:, :, abs(dx), abs(dy)]
            
            # Remove the co-occurrence matrix for the previous window column
            if j >= offset*2:
                for k in range(w):
                    y = j - offset*2 + k
                    x = i - offset
                    i1 = image[x, y]
                    for l in range(k+1, w):
                        y1 = j - offset*2 + l
                        i2 = image[x, y1]
                        co_occurs[i1, i2, k, l] -= 1
                        co_occurs[i2, i1, l, k] -= 1

    for i in range(len(distances)):
        for j in range(len(angles)):
            glcm[:,:,i, j] = glcm[:,:,i,j]/np.sum(glcm[:,:,i,j])

    return glcm                            # normalised glcm

def GLCM(image, distances, angles, levels):
    height, width = image.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)))

    for i in range(height):
        for j in range(width):
            for d, distance in enumerate(distances):
                for a, angle in enumerate(angles):
                    dx = round(distance * np.cos(angle))
                    dy = round(distance * np.sin(angle))
                    x1 = i + dx
                    y1 = j + dy
                    x2 = i - dx
                    y2 = j - dy
                    if x1 < 0 or x1 >= height or y1 < 0 or y1 >= width:
                        continue
                    if x2 < 0 or x2 >= height or y2 < 0 or y2 >= width:
                        continue
                    i1 = image[x1, y1]
                    i2 = image[x2, y2]
                    glcm[i1, i2, d, a] += 1
                    glcm[i2, i1, d, a] += 1
    for i in range(len(distances)):
        for j in range(len(angles)):
            glcm[:,:,i, j] = glcm[:,:,i,j]/np.sum(glcm[:,:,i,j])

    return glcm

def Algorithm(image_path, num_clusters, outpath):
    # Load the input image and convert it to grayscale
    image = Image.open(image_path)
    img = image.convert("L")
    img = np.array(img)

    # Selecting the window of the image and doing all computation on that particular image
    if img.shape[0]<=250 or img.shape[1] <=250 :
        gray = img
    else:
        gray = img[:250, :250]
    

    # Setting up parameters for the GLCM mattrix computation
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]


    ''' Computing GLCM of each window by sliding window and extracting features of each window respectively
    '''
    w  = 16                     # Window size
    # w = (w//2)*2

    levels = 256                # Gray levels
    rows, cols = gray.shape
    feature1 = np.zeros((rows-w, cols-w, len(distances)* len(angles)))
    feature2 = np.zeros((rows-w, cols-w, len(distances)* len(angles)))
    feature3 = np.zeros((rows-w, cols-w, len(distances)* len(angles)))

    for i in range(w//2, rows-w//2):
        for j in range(w//2, cols-w//2):
            window = gray[i-w//2:i+w//2+1, j-w//2:j+w//2+1]         # Defining the window to compute glcm
            glcm = fastglcm(window, distances=distances, angles=angles, levels=levels)

            
            f1 = []           # Corresponds to contrast
            f2 = []           # Corresponds to homogeneity
            f3 = []           # Corresponds to entropy

            for d in range(len(distances)):
                for a in range(len(angles)):

                    glcm_d_a = glcm[:,:, d, a]              # GLCM matrix at particular angle and distance

                    # Computing the features with the corresponding formulas mentioned in the report and adding it to the feature array

                    f1.append(np.sum(np.square(np.arange(levels)[:, np.newaxis] - np.arange(levels)[np.newaxis, :]) * glcm_d_a))        # Contrast
                    f2.append(np.sum(glcm_d_a / (1 + np.square(np.arange(levels)[:, np.newaxis] - np.arange(levels)[np.newaxis, :]))))  # Homogeneity
                    f3.append(-np.sum(glcm_d_a * np.log2(glcm_d_a + (glcm_d_a == 0))))                                                  # Entropy


            feature1[i-w//2, j-w//2, :] = np.reshape(f1, (len(angles)*len(distances), ))
            feature2[i-w//2, j-w//2, :] = np.reshape(f2, (len(angles)*len(distances), ))
            feature3[i-w//2, j-w//2, :] = np.reshape(f3, (len(angles)*len(distances), ))

 
    ''' combining a particular feature computed at different angles and distances for each pixel location '''
    ''' resulting in 3 channels of the output image '''

    c1 = np.mean(feature1, axis=2)
    c1 = ((c1 - np.min(c1))/(np.max(c1)-np.min(c1))).reshape(c1.shape[0], c1.shape[1], 1)       # Normalising the feature values obtained, 
                                                                                                # for them to be reasonable to give to KMeans
    c2 = np.mean(feature2, axis=2)
    c2 = ((c2 - np.min(c2))/(np.max(c2)-np.min(c2))).reshape(c2.shape[0], c2.shape[1], 1)

    c3 = np.mean(feature3, axis=2)
    c3 = ((c3 - np.min(c3))/(np.max(c3)-np.min(c3))).reshape(c3.shape[0], c3.shape[1], 1)


    ''' Merging the feature/channels to form an image of the same dimensions as original '''
    feature = np.concatenate([c1, c2], axis=-1 )
    feature = np.concatenate([feature, c3], axis=-1)
    #plt.imshow(feature)
    print("Features computed...")
    

    ''' Classification by K-means clustering algorithm '''
    k = num_clusters            # Number of clusters
    kmeans_feature = feature.reshape((feature.shape[0]*feature.shape[1], feature.shape[2]))
    kmeans_feature = rescale(kmeans_feature, gray)

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(kmeans_feature)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
    # Plotting the original image.
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the feature image
    ax[1].imshow(labels.reshape(gray.shape[0],gray.shape[1]))
    ax[1].set_title('Result image with k ={}'.format(k))
    ax[1].axis('off')
    
    plt.subplots_adjust()
    plt.savefig(outpath)

    plt.show()

