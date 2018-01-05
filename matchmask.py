### This library is code for matching the well mask to droplets images in order to
### identify droplets that are in the same well.

import numpy as np
from scipy.ndimage.filters import convolve
from skimage.feature import register_translation
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as label_bw

####################################
############ ROTATION ##############

def fft(channel):
    f = np.fft.fft2(channel)
    f = np.fft.fftshift(f)
    return np.absolute(f)

def clip_image(image, size=(101,101)):
    sz = image.shape

    slices = []
    for axis in range(len(image.shape)):
        diff = sz[axis]-size[axis]
        slices.append(slice(diff/2,-diff/2))

    return image[slices]

def compute_f(image,f):
    sz = image.shape
    x = np.arange(sz[1])
    f_i = {}
    f_i1 = {}
    for i in x:
        f_i[i] = f(i-0.5)
        f_i1[i] = f(i-0.5+1)

    out = np.zeros(image.shape)
    for ix in range(sz[1]):
        for jy in range(sz[0]):
            if ((f_i[ix] <= (jy-0.5)) & (f_i1[ix] > (jy-0.5))) | ((f_i[ix] >= (jy-0.5)) & (f_i1[ix] < (jy+0.5))):
                out[jy,ix] = 1

    return out

def theta_transform(image):
    theta = np.hstack((np.linspace(0,np.pi/2,90), np.linspace(np.pi/2,np.pi,90)))
    coef = np.tan(theta)

    sz = image.shape[0]
    shift = sz/2+1

    m = []
    for th in coef:
        idx = compute_f(image,lambda x: th*(x-shift)+shift)
        m.append((image[idx==1]).mean())

    return theta, np.asarray(m)

def rotate(arr, theta):
    rotation_matrix = np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return np.dot(arr,rotation_matrix.T)

#############################################
######### SLICE MASK ########################

def slice_mask(config,rotation):
    FOV_size = config['image']['size']
    start_image = config['image']['well_start_image']
    overlap = config['image']['overlap']

    # Top left shift
    row_shift = config['well_mask']['well_start_xy'][1]-config['image']['well_start_xy'][1]
    col_shift = config['well_mask']['well_start_xy'][0]-config['image']['well_start_xy'][0]

    topleft = np.asarray([col_shift, row_shift])

    shiftx = lambda x,y: (x-start_image[0])*(FOV_size)*(1-overlap)
    shifty = lambda x,y: (y-start_image[1])*FOV_size*(1-overlap)+np.round(np.tan(rotation)*shiftx(x,y))
    slicer = lambda x,y,error: [slice(shifty(x,y)+topleft[1]-error, shifty(x,y)+FOV_size+topleft[1]+error),slice(shiftx(x,y)+topleft[0]-error, shiftx(x,y)+FOV_size+topleft[0]+error)]
    return slicer

#############################################
############ SYNTHESIZE DROPLET IMAGES ######

def find_range(pos,radius=10):
    rng = np.vstack((pos.min(axis=0),pos.max(axis=0)))
    rng[0] = rng[0]-2*radius
    rng[1] = rng[1]+2*radius
    return np.round(rng)

def shift_to_zero(pos,radius=10):
    pos_ = pos.copy()
    pos_[:,0]= pos_[:,0]-pos[:,0].min()+2*radius
    pos_[:,1]= pos_[:,1]-pos[:,1].min()+2*radius
    return pos_

def initialize_image(pos):
    shifted = shift_to_zero(pos)
    rng = find_range(shifted)

    img = np.zeros(rng[1][[1,0]].astype('int'),'bool')

    return img, shifted

def flip_center_bits(img,shifted):
    img_ = img.copy()
    p = (np.round(shifted)).astype('int')

    img_[p[:,1],p[:,0]]=1

    return img_

def disk_kernel(size,radius):
    k = np.zeros(size,'bool')
    center = np.asarray(size).astype('float')/2

    for i in range(size[0]):
        for j in range(size[1]):
            if np.sqrt(((np.asarray((i,j))-center)**2).sum())<radius:
                k[i,j]=True
    return k

def synthesize_image(pos):
    img, shifted = initialize_image(pos)
    img = flip_center_bits(img, shifted)
    syn_droplets_img = convolve(img,disk_kernel((30,30),10))
    return syn_droplets_img, shifted


###########################################################
##################### REGISTER SYN IMAGE TO MASK #########

def pad_images_to_same(imgs_):
    # Input tuple of images
    shifts = []
    imgs = [item for item in imgs_]
    for axis in range(len(imgs[0].shape)):
        maxx = np.asarray([i.shape[axis] for i in imgs]).max()

        for iImg in range(len(imgs)):
            img_ = imgs[iImg]
            diff = maxx-img_.shape[axis]

            imgs[iImg] = pad_axis(img_,axis,diff)
            shifts.append(diff/2)

    return tuple(imgs), shifts

def pad_axis(img,axis,pad_width):
    if pad_width != 0:
        if axis == 0:
            new_img = np.zeros((np.asarray(img.shape[0])+pad_width,img.shape[1]))
            new_img[pad_width/2:-pad_width/2,:]=img
        else:
            new_img = np.zeros((img.shape[0],np.asarray(img.shape[1])+pad_width))
            new_img[:,pad_width/2:-pad_width/2]=img
        return new_img
    else:
        return img

###########################################################
##################### DETECT WELLS ON EDGES #########

def edge_mask(xy,mask):
    emask = np.zeros(mask.shape)
    emask[:np.percentile(xy[:,1],1),:]=1
    emask[np.percentile(xy[:,1],99):,:]=1
    emask[:,:np.percentile(xy[:,0],1)]=1
    emask[:,np.percentile(xy[:,0],99):]=1
    return emask
