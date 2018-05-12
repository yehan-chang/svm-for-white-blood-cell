"""
### Items included in Deliverables
* CA12 Report
* Dataset for classification problem
* Python Source code

## Members
* Chang Ye Han
* Chee Jiawei
* Chua Zhen Liang Desmond
* Ganesh Kumar
* Goh Yu Chen
"""

import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA # Not use after refinement
from skimage.transform import rescale # Not use after refinement
from sklearn.metrics import confusion_matrix

EOSINOPHIL_folders = ['../images/EOSINOPHIL/']
LYMPHOCYTE_folders = ['../images/LYMPHOCYTE/']
MONOCYTE_folders = ['../images/MONOCYTE/']
NEUTROPHIL_folders = ['../images/NEUTROPHIL/']

test_img_loc = '../images/EOSINOPHIL/_0_207.jpeg'
feature_image = cv2.imread(test_img_loc)
feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)
plt.imshow(feature_image)
plt.show()

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 44 # HOG pixels per cell default 8
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions default 32
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

SIZE = 2478 # Number of images uses per classification for training and testing
rescaleFactor = 3.0 # Rescale factor to reduce memory usage
Cparameter= 0.001

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


feature_image = cv2.imread(test_img_loc)
feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)

hog_features, hog_image = get_hog_features(feature_image[:,:,0], orient, 
                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)
plt.imshow(hog_image)
plt.title('HOG Image ' + str(pix_per_cell) + ' Pixel-per-cell')
plt.show()


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Plot features
feature_image = cv2.imread(test_img_loc)
feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)
feature_vec = bin_spatial(feature_image, size=spatial_size)
plt.plot(feature_vec)
plt.title('Spatially Binned Features COLOR_RGB2HSV')
plt.show()


# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

feature_image = cv2.imread(test_img_loc)
feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)
hist_features = color_hist(feature_image, nbins=hist_bins)
plt.hist(hist_features)
plt.title('Color Histogram Features COLOR_RGB2HSV')
plt.show()


# Define a function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hog_features=None):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      

    # Rescale function to reduce the memory usage    
#    feature_image = rescale(feature_image, 1.0 / rescaleFactor)
    
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if True: #hog_features is None:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        img = cv2.imread(file)
        
        file_features = single_img_features(img, color_space, spatial_size,
            hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
            spatial_feat, hist_feat, hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features



def prepare_images_for_processing(EOSINOPHIL_folders, LYMPHOCYTE_folders, MONOCYTE_folders, NEUTROPHIL_folders, image_type):
    EOSINOPHIL_array = []
    for folder in EOSINOPHIL_folders:
        EOSINOPHIL_array += glob.glob(folder +'/*.' + image_type)
        
    LYMPHOCYTE_array = []
    for folder in LYMPHOCYTE_folders:
        LYMPHOCYTE_array += glob.glob(folder +'/*.' + image_type)
        
    MONOCYTE_array = []
    for folder in MONOCYTE_folders:
        MONOCYTE_array += glob.glob(folder +'/*.' + image_type)
        
    NEUTROPHIL_array = []
    for folder in NEUTROPHIL_folders:
        NEUTROPHIL_array += glob.glob(folder +'/*.' + image_type)
        
    # Keep distribution even
    EOSINOPHIL_array = EOSINOPHIL_array[:SIZE]
    LYMPHOCYTE_array = LYMPHOCYTE_array[:SIZE]
    MONOCYTE_array = MONOCYTE_array[:SIZE]
    NEUTROPHIL_array = NEUTROPHIL_array[:SIZE]
    
    print ("Read EOSINOPHIL")
    EOSINOPHIL_features = extract_features(EOSINOPHIL_array, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
    
    print ("Read LYMPHOCYTE")
    LYMPHOCYTE_features = extract_features(LYMPHOCYTE_array, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
    
    print ("Read MONOCYTE")
    MONOCYTE_features = extract_features(MONOCYTE_array, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
    
    print ("Read NEUTROPHIL")
    NEUTROPHIL_features = extract_features(NEUTROPHIL_array, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
    
    print ("Stack")
    X = np.vstack((EOSINOPHIL_features, LYMPHOCYTE_features, MONOCYTE_features, NEUTROPHIL_features)).astype(np.float64)   
   
    print ("Fit")                     
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    print ("Scaler")
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    print ("Define labels")
    # Define the labels vector
    list0 = [0] * SIZE
    list1 = [1] * SIZE
    list2 = [2] * SIZE
    list3 = [3] * SIZE
    y = list0 + list1 + list2 + list3
    
    return scaled_X, np.array(y), X_scaler

t=time.time()

#Read the image
scaled_X, y, X_scaler = prepare_images_for_processing(EOSINOPHIL_folders, LYMPHOCYTE_folders, MONOCYTE_folders, NEUTROPHIL_folders, "jpeg")

# PCA model not in used
#print ("PCA start")
#my_model = PCA(n_components=0.9999, svd_solver='full')
#scaled_X = my_model.fit_transform(scaled_X)  
#print ("PCA end")

#Shuffle to randomise the data
scaled_X, y = shuffle(scaled_X, y, random_state=256)

#Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC(C = Cparameter)

tstart=time.time()
# Check the training time for the SVC
svc.fit(X_train, y_train)
t2 = time.time()
pred = svc.predict(X_test)
print(round(t2-tstart, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
       
# Check the score of the train   
print('Test Accuracy of train = ', round(svc.score(X_train, y_train), 4))
t3=time.time()
print(round(t3-t, 2), 'Seconds to END...')


#Run this line to get the confusion matrix of the result
confusion_matrix(y_test, pred)
