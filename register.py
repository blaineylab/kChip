import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
import io as kchip_io
from skimage.feature import register_translation

def global_coordinates(config,df):
    '''
    Compute global coordinates from local image coordinates in supplied dataFrame.
    Inputs:
        - config dictionary
        - df, dataFrame
    Outputs:
        - df with added global coordinates added in new columns
    '''

    px_to_global = lambda (image_num, pixel_num): (image_num-1)*config['image']['size']*(1-config['image']['overlap']) + pixel_num

    Global_X = df.apply(lambda row: px_to_global((row['IndexX'],row['ImageX'])),axis=1)
    Global_Y = df.apply(lambda row: px_to_global((row['IndexY'],row['ImageY'])),axis=1)

    df_ = df.copy()
    df_['GlobalX'] = Global_X
    df_['GlobalY'] = Global_Y

    return df_

def register(config,img_tuple,t='premerge',t2='t0'):
    '''Register a translation between a given pre- and post-merge image tuple
    Inputs:
        - config, the config file dictionary
        - img_tuple: a tuple of image pathnames for pre- and post-merge images
        - t: the pre-merge timepoint name
        - t2: the post-merge timepoint name
    Returns:
        - shift, a 1 x 2 numpy array to be "translated/subtracted" from coordinates in the second image in tuple
    '''

    # Read images
    pre_img = kchip_io.read(x=img_tuple[0],y=img_tuple[1],t=t)
    post_img = kchip_io.read(x=img_tuple[0],y=img_tuple[1],t=t2)

    slices = np.delete(np.arange(pre_img.shape[2]),config['image']['bugs'])

    shift,error,diffphase = register_translation(pre_img[:,:,slices], post_img[:,:,slices])
    return shift[:-1]

def select_wells_in_image(df,x_,y_):
    ''' Return wells in dataFrame in the requested image.
    Inputs:
        - df, dataFrame, as returned by drops.preimage_to_droplets or drops.postimage_to_bugs
        - x_, integer of current image
        - y_, integer of current image
    Outputs:
        - (n x 2) numpy array of global coordinates of wells in image
        - (1 x n) numpy array of indices of wells selected in supplied df
    '''
    idx = df.index[(df['IndexX']==x_) & (df['IndexY']==y_)]
    df_ = df.loc[idx]
    return np.vstack((df_['GlobalX'],df_['GlobalY'])).T, idx

def overlapping_images(df,x_,y_):
    ''' Return overlapping images based on images available in supplied dataFrame.
    Inputs:
        - dataFrame, as returned by drops.preimage_to_droplets or drops.postimage_to_bugs
        - x_, integer of current image
        - y_, integer of current image
    Outputs:
        - list of tuples of (x,y) images neighboring input x, y
    '''
    maxX = df['IndexX'].max()
    maxY = df['IndexY'].max()

    if x_ == maxX:
        listx = [x_]
    else:
        listx = [x_,x_+1]

    if y_ == maxY:
        listy = [y_]
    else:
        listy = [y_,y_+1]

    return [(tx,ty) for tx in listx for ty in listy]

def match_wells(w1,w2):
    ''' Return assignment in w2 that minimize distance to each point in w1
    Inputs:
        - w1, (n x 2 numpy array), pre-merge coordinates
        - w2, (m x 2 numpy array), post-merge coordinates
    Returns:
        - pre, indices in w1 (should stay 0 to n)
        - post, indices in w2 that minimize distance to corresponding points in w1
        - dist, distance of minimum
    '''
    distance = cdist(w1,w2)
    pre = np.arange(w1.shape[0])
    post = np.argmin(distance,axis=1)
    dist = distance[pre,post]

    return pre, post, dist

def assign_wells(wells,post_wells,shift=0,threshold=150):
    ''' Assign wells between pre- and post- merge images.
    Inputs:
        - droplets (pandas dataFrame), output by drops.preimage_to_droplets
        - end_droplets (pandas dataFrame), output by drops.postimage_to_bugs
        - (optional) shift (2 x 1 numpy array), translation to be applied to coordinates in end_droplets
        - (optional) threshold (integer, units of global coordinates), a distance threshold for deleting rows of assignments
    '''

    # Create list for storing well assignments for wells in each image
    well_assignments_list = []

    for iX in wells['IndexX'].unique():

        for iY in  wells['IndexY'].unique():

            # Pull out the pre-merge coordinates from droplets dataframe
            premerge, preidx = select_wells_in_image(wells,iX,iY)

            # Pull out the post-merge coordinates, and translate the postmerge images according to the input shift
            postmerge_list = [select_wells_in_image(post_wells,item[0],item[1]) for item in overlapping_images(post_wells,iX,iY)]
            postmerge = np.vstack(tuple(item[0] for item in postmerge_list))+shift
            postidx = np.hstack(tuple(item[1] for item in postmerge_list))

            # match the wells from pre, and post merge
            match_pre, match_post, match_dist = match_wells(premerge,postmerge)

            well_assignments_list.append(np.asarray([preidx[match_pre], postidx[match_post], match_dist]))

    # collapse well assignments list into single ndarray
    well_assignments = np.hstack(well_assignments_list).T

    # Remove if below threshold
    well_assignments = np.delete(well_assignments,(well_assignments[:,2] > threshold).nonzero(),axis=0)

    return well_assignments

def return_conflicts(array_):
    ''' Searches array for non-unique values, and then returns list of indices.
    Inputs:
        - array_, a ***FLAT*** numpy array
    Returns:
        - list of arrays of indices that share same value
    '''

    uniq, uniq_counts = np.unique(array_,return_counts=True)
    found = (uniq_counts>1).nonzero()
    conflicts = uniq[found]

    idx = [(array_==i).nonzero()[0] for i in conflicts]

    return idx

def resolve_conflicts(well_assignments):
    '''
    Resolve conflicts within the pre- and post- merge well assignments. Returns well_assignments with
    conflicting assignments set to -1.
    Inputs:
        - well_assignments (n x 3 numpy array)
            - column 1: indices of rows within droplets dataFrame
            - column 2: indices of rows within end_droplets dataFrame
            - column 3: distance between well in col1 and well in col2 to use for resolving conflicts
    Returns:
        - assignments, well_assignments with conflicts set to -1, except for min distance
        - removed_index, the indices of rows set to -1 in returned assignments
    '''
    assignments = well_assignments.copy()

    conflict_index = return_conflicts(assignments[:,1])

    removed_index = []
    for points in conflict_index:
        change_idx = (assignments[points,2]!=np.min(assignments[points,2]))
        assignments[points[change_idx],1]=-1
        removed_index.append(points[change_idx])

    if len(removed_index)>0:
        rmv = np.hstack(removed_index)
    else:
        rmv = np.asarray([])

    return assignments, rmv
