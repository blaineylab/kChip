# IO utils

from os import listdir, path
import numpy as np
import yaml
import re
from skimage import io
import pandas as pd
from ast import literal_eval

def list_images(base_path, filename):
    '''Returns list of all images in base_path that start with filename.

        Inputs:
            - base_path (str), the directory
            - filename (str), start of filename

        Returns:
            - list of all image files found in directory (list of strings)
    '''

    all_files = listdir(base_path)
    selected = [a for a in all_files if (filename in a) and ('.tif' in a)]

    exp = re.compile(filename+'_(\d+)_(\d+).tif')
    idx = list_index(selected,exp)

    return selected, idx

def list_index(image_list,exp):
    ''' Returns array of x, y indexes found in image_list for supplied regex.

        Inputs:
            - image_list (list of str), from list_images
            - exp (re.compile object), regex to match

        Returns:
            - (n x 2) array
    '''
    idx = []
    for item in image_list:
        t = exp.match(item)
        if t:
            idx.append(np.asarray([int(a) for a in t.groups()]))

    idx = np.vstack(idx)

    return idx

def read(config,x,y,t, number=5, ret=(0,1,2,3,4)):
    ''' Read in image corresponding to position (x,y) and t (string).

    Inputs:
        - config, the config dictionary
        - x (int), the x index
        - y (int), the y index
        - t (str), timepoint corresponding to config name, e.g. 'premerge', 't0', 't1'
        - (optional) number (int), the number of channels in the image (default 4), to detect issues
        - (optional) ret (tuple of int), the slices to return, default (0,1,2,3)
    Outputs:
        - a stack of images as 3 dimensional numpy array (slices are axis=2)
        
    img.shape will be (slices,pixels in x, pixels in y) (example: 5,2048,2048)
    transpose changes it to (pixels in x, pixels in y, slices) (example: 2048,2048,5)
    '''
    fname = path.join(config['image']['base_path'],config['image']['names'][t]+'_'+str(x)+'_'+str(y)+'.tif')
    img = io.imread(fname)
    
    if 'background_image' in config['image'].keys():
        bkgdname = path.join(config['image']['base_path'],config['image']['background_image']+'.tif')
        bkgd = io.imread(bkgdname)
        img = img.astype('float64')-bkgd.astype('float64')
        img[img<0] = 0
        
    if 'flatfield_image' in config['image'].keys():
        ff_name = path.join(config['image']['base_path'],config['image']['flatfield_image']+'.tif')
        ff_img = io.imread(ff_name)
        ff_img = ff_img.astype('float64')
        for i in range (0,5):
            ff_img[i] = ff_img[i]/ff_img[i].max()
        img = img/ff_img
    
    if 'rescale' in config['image'].keys():
        rescale = config['image']['rescale']
    else:
        rescale = np.ones(number)

    # a) Transpose if necessary to (x,y,z)
    if img.shape[2] > img.shape[0]:
        img_ = img.transpose((1,2,0))
    else:
        img_ = img

    # b) return slices if necessary
    if img_.shape[2] > number:
        return img_[:,:,ret]*rescale
    else:
        return img_*rescale


def read_excel_barcodes(config):
    ''' Read in excel barcodes and returns dictionary label -> barcode '''

    barcodes = pd.read_excel(config['barcodes']['path'],sheetname='Barcodes')
    labels = pd.read_excel(config['barcodes']['path'],sheetname='Labels')

    d = dict(zip(labels.values.reshape(-1),barcodes.values.reshape(-1)))

    for item in d.keys():
        if item == item:
            d[item] = literal_eval(d[item])
        else:
            del d[item]

    return d
