from skimage.filters import sobel
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage.filters import convolve

from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##################### IDENTIFY DROPLETS ###################

def threshold_image(image,threshold=0.15,show=0):
    '''Thresholds image based on input
    Inputs:
    - n x m numpy array,uint16
    Outputs:
    - n x m numpy array, boolean
    '''
    bkg = image.mean()
    s = image.std()

    mask = image > bkg+threshold*s

    if show:
        plt.imshow(mask)
        plt.axis('off');

    return mask

def find_droplets(config,image,threshold=0.53,show=1):
    '''Apply Hough Transform to find droplets in the image
    Inputs:
    - image (n x m uint16 array), a 2D grayscale image
    - threshold (0-1), threshold for hough transform accumulators (Default = 0.85)
    - show (=0 or =1), to show output
    Returns:
    - k x 2 integer array, corresponding to centers of droplets found
    '''

    # 1) Find mask based on thresholding summed image
    mask = threshold_image(image,threshold=0.15)
    edges = sobel(mask)

    # 2) Hough transform to find droplets
    # hough_radii = np.arange(8,11)
    # Droplet size range should be radius range of 52 um to 80 um
    hough_radii = np.arange(52./config['image']['pixel_size'],80/config['image']['pixel_size'])
    hough_res = hough_circle(edges,hough_radii)
    # 2b) Take the maximum to avoid redundancies at different radii
    hough_res = np.max(hough_res,axis=0)

    # 2c) Find the peaks in the accumulators
    accum, cx, cy, rad = hough_circle_peaks([hough_res],[10.],threshold=threshold,min_xdistance=15,min_ydistance=15)

    if show:
        print 'Found '+str(len(cx))+' Droplets'

#         fig, axes = plt.subplots(figsize=(10,10))
#         axes.imshow(image)
#         plt.axis('off');

#         for center_y, center_x, radius in zip(cy, cx, rad):
#             circy, circx = draw_circle(center_y, center_x, radius)
#             axes.plot(circx,circy,'r')

    return pd.DataFrame(data=np.vstack((cx,cy)).T,columns=['ImageX','ImageY'])

def draw_circle(x,y,rad,density = 25.):
    '''Return the coordinates for drawing circle, for use with a plotting tool
    Inputs
    - x, (int, float) x coordinate
    - y, (int, float) y coordinate
    - radius (int, float), the radius
    - (optional) density, the number of points to return over 0, 2pi range
    '''
    theta = np.arange(0,2*np.pi,2*np.pi/density)
    return rad*np.cos(theta)+x, rad*np.sin(theta)+y

def remove_overlap(df,config,show=0):
    ''' Remove droplets found in regions of image that overlap.
    Inputs:
        - df, droplets dataFrame
        - config, the config dictionary read from yaml file
    Returns:
        - copy of dataframe, with dropped rows
    '''
    maxX = df['IndexX'].max()
    maxY = df['IndexY'].max()
    overlap = (1-config['image']['overlap'])*config['image']['size']

    rmv_index = ((df.ImageX > overlap) & ~(df.IndexX==maxX)) | ((df.ImageY > overlap) & ~(df.IndexY==maxY))

    if show:
        print 'Removed: ' + str(np.sum(rmv_index)) + ' wells from dataFrame due to overlap in images.'

    return df.drop(df.index[rmv_index]).reset_index(drop=True)

################### DYE ESTIMATION ####################

def local_average(image,filter_size=5,show=0):
    '''Compute local averages by convolution for droplet color estimation
    Inputs
    - image (2/3-d array, uint16)
    - filter_size (odd integer), the size of the window to use
    Outputs
    - filtered_image (2/3-d array, same size as input)
    '''
    f = 1./filter_size**2*np.ones((filter_size,filter_size))

    if len(image.shape) == 3:
        maxSlice = image.shape[2]
    else:
        maxSlice = 1

    filtered_image = np.zeros(image.shape)
    for iSlice in range(maxSlice):
        filtered_image[:,:,iSlice] = convolve(image[:,:,iSlice],f)

    if show:
        plt.figure(figsize=(20,20))
        plt.imshow(filtered_image.sum(axis=2))

    return filtered_image


############# POST-MERGE IMAGES ######################


def mask_post_merge(image,area_threshold=100):
    '''Mask the wells in post-merge image and filter for area
    Inputs:
    - image (2d array)
    - area_threshold, the minimum area of connected components to return
    Outputs
    - end_mask_filtered (boolean, same dimensions as image)
    - labeled (ndarray of dtype int)
    '''

    # Mask image
    end_mask = threshold_image(image)

    # Remove small components
    end_mask_filtered = remove_small_objects(end_mask,area_threshold)
    labeled = label(end_mask_filtered)
    return labeled

def post_img_to_wells(config,post_img):
    labeled = mask_post_merge(post_img[:,:,config['image']['dyes']].sum(axis=2))

    # Calculate region properties
    m = regionprops(labeled,post_img[:,:,config['image']['bugs']],cache=True)

    if len(m):
        return pd.DataFrame(data=np.hstack((extract(m,'area'),extract(m,'centroid'),extract(m,'mean_intensity'))),columns=['Area','ImageY','ImageX','Intensity'])
    else:
        return pd.DataFrame(columns=['Area','ImageY','ImageX','Intensity'])

def extract(measurements,prop):
    ''' Extract a given property from regionprops measurements array '''
    try:
        sz = len(measurements[0][prop])
    except:
        sz = 1
    return np.asarray(map(lambda in_: in_[prop],measurements)).reshape(len(measurements),sz)
