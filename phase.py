import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from droplets import threshold_image
import io as kchip_io

from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter

from skimage.morphology import remove_small_objects,erosion, disk
from skimage.measure import regionprops,label
import skimage.filters as filters


def range_filter(image,disk_size=2):
    ''' Apply range filter to image. This returns the max-min within a given window at each pixel.
    Inputs:
        image: m x n, numpy array
    Outputs:
        image_range: range filtered image
    '''

    selem = disk(disk_size)

    image_min = minimum_filter(image,footprint=selem)
    image_max = maximum_filter(image,footprint=selem)

    image_range = image_max-image_min

    return image_range

def rescale(img):
    return (img-img.min()).astype('float')/(img.max()-img.min())

def phase_high_pass(config,phase):
    ''' Sharpen image by computing a high_pass filter.
    Inputs:
        - config, the config dictionary
        - phase, the phase image
    Outputs:
        - phase_dog, the high pass filtered image
    '''

    phase_bkg_sigma = config['image']['phase']['bkg_sigma']
    phase_bkg = filters.gaussian(phase,sigma=phase_bkg_sigma,preserve_range=True)
    phase_dog = phase-phase_bkg
    phase_dog = (65535*rescale(phase_dog)).astype('uint16')
    return phase_dog

def split_channels(config, image):
    ''' Split the image into gfp, dye, and phase channels.
    Inputs:
    - config dictionary
    - 3d np array
    Outputs:
    - gfp, image
    - dyes, image (sum of dye channels)
    - phase, image
    '''
    gfp = image[:,:,config['image']['phase']['bugs']]
    dyes = image[:,:,config['image']['phase']['dyes']].sum(axis=2)
    phase = image[:,:,config['image']['phase']['phase']]
    return gfp, dyes, phase

def phase_mask(config,phase):
    ''' Compute mask from phase image, to eliminate droplet edges and posts
    Inputs:
    - config, config dictionary
    - phase, the phase image
    Outputs:
    - mask, the phase mask
    '''
    return threshold_image(gaussian_filter(phase,config['image']['phase']['mask_sigma']))==0

def dye_mask(config,dyes):
    ''' Segment droplets from dyes image.
    Inputs:
        - dyes, image (1 slice) computed from sum of dye channels
    Outputs:
        -dye_mask_label, the label image of segmented droplets
    '''

    # blur image
    dyes_blurred = gaussian_filter(dyes,3)
    # threshold image
    dyes_mask_all = threshold_image(dyes_blurred)
    # remove small objects
    small_object_size = 100*(6.5/config['image']['phase']['pixel_size'])**2

    dyes_mask = remove_small_objects(dyes_mask_all,small_object_size)

    #  Eroding mask to exclude periphery - disk size can be changed
    selem = disk(20);
    dyes_mask_erode = erosion(dyes_mask,selem=selem);

    dyes_mask_label = label(dyes_mask_erode)

    return dyes_mask_label

def initialize_phase(config,timepoint):
    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names'][timepoint])

    phase_df = []
    for xy in image_idx:
        print 'Now analyzing: ', xy[0],',',xy[1]

        t = kchip_io.read(config,x=xy[0],y=xy[1],t=timepoint,number=5)
        gfp, dyes, phase = split_channels(config,t)

        mask = dye_mask(config,dyes)*phase_mask(config,phase)
        phase_signal = range_filter(phase_high_pass(config,phase))

        phase_props = regionprops(mask,phase_signal)
        gfp_props = regionprops(mask,gfp)

        one = np.asarray([p['centroid'] for p in phase_props])
        two = np.asarray([p['mean_intensity'] for p in phase_props])[:,np.newaxis]
        three = np.asarray([p['mean_intensity'] for p in gfp_props])[:,np.newaxis]

        data = np.hstack((one, two, three))
        phase_df_ = pd.DataFrame(data=data,columns=['ImageX','ImageY','pGFP','Phase'])
        phase_df_['IndexX']=xy[0]
        phase_df_['IndexY']=xy[1]
        phase_df.append(phase_df_)

    phase_df = pd.concat(phase_df)
    phase_df.reset_index(inplace=True,drop=True)
    return phase_df
