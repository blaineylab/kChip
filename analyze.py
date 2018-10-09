import pandas as pd
import numpy as np
import skimage.io as io
from skimage.transform import rescale
import matplotlib.pyplot as plt

#kchip imports
import droplets as drop
import matchmask
import io as kchip_io
import cluster as cluster
import register as reg

####################### CREATE DROPLETS DATAFRAME ###################

def initialize_droplets(config):
    '''Initialize the droplets DataFrame by finding droplets in all images.
    Inputs:
        - config, the config dictionary created from config
    Outputs:
        - droplets (DataFrame), with Image index and position, and RGB data
     '''

    # Determine image list
    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names']['premerge'])

    # initialize list of droplets dataframes from each image
    droplets = []

    # initialize average f_img, which later averages all the fourier transforms
    # to detect the rotation
    f_img = 0

    for xy in image_idx:
        print 'Creating droplets from: '+str(xy[0])+','+str(xy[1])

        # Read in image
        img = kchip_io.read(config, x=xy[0],y=xy[1],t='premerge')

        # Locate droplets and store in temporary dataframe, then append to list of dataframes
        droplets_ = drop.find_droplets(config,img.sum(axis=2),threshold=0.53,show=0)
        droplets_.insert(0,'IndexY',xy[1])
        droplets_.insert(0,'IndexX',xy[0])

        # Pull out the RGB value for each droplet and store in dataframe

        # Compute local average
        dyes = drop.local_average(img[:,:,config['image']['dyes']])
        to_add = pd.DataFrame(dyes[droplets_['ImageY'],droplets_['ImageX']])

        # Fix a bug where if there is only 1 droplet detected it is wrong orientaiton
        if to_add.shape[1] != 3:
            to_add = to_add.T

        droplets_[['R','G','B']]= to_add

        # append dataframe
        droplets.append(droplets_)

        # fft the image, clip, and store update average
        f_img += matchmask.clip_image(matchmask.fft(img.mean(axis=2))).astype('float')/len(image_idx)

    # Concatenate all the droplets_ dataframes
    droplets = pd.concat(droplets).reset_index(drop=True)

    # Correct the rotation
    droplets, rotation_theta = apply_rotation(droplets, f_img)

    return droplets, rotation_theta

def apply_rotation(droplets,f_img):
    ''' Compute the rotation for aligning to well mask.
    - Integrate the fourier transform along lines through the center of the image, as function of the line angle (theta).The rotation needed to fix the image should maximize this value.
    - Find the theta that yields the maximum

    Then compute the rotation matrix, and update the droplets DataFrame with rotated coordinates.

    Inputs:
        - droplets (the droplets DataFrame)
        - f_img (the average fourier transform)

    Outputs:
        - droplets_, a copy of the droplets DataFrame with rotated coordinates added (RX, RY)
        - rotation, the theta calculated

    '''

    f_img_ = f_img.copy()
    droplets_ = droplets.copy()

    # Delete the 51,51 coordinate
    f_img_[51,51]=0

    # Integrate fourier transform along lines through the center of the image
    theta_, score_  = matchmask.theta_transform(f_img_)

    # Find the theta that yields the maximum value
    rotation = -theta_[score_==score_.max()][0]+np.pi/2

    # Since some well arrays are symmetric over a 90-degree rotation, sometimes this is wrong and we have to correct it
    if rotation >= np.pi/4:
        rotation = rotation-np.pi/2

    rotated_coordinates =matchmask.rotate(droplets[['ImageX','ImageY']].values,rotation)

    droplets_['RX']=rotated_coordinates[:,0]
    droplets_['RY']=rotated_coordinates[:,1]

    return droplets_, rotation

######################################## FIT TO MASK #########################

def initialize_well_mask(config, rotation_theta):
    ''' Initialize the well_mask used to identify droplets in the same well.
    Inputs:
        - config, the config dictionary created from config
        - rotation_theta, the angle of rotation calculated in apply_rotation (called from initialize_droplets)
    Outputs:
        - well_mask, the well mask image
        - mask_xy, a function that inputs x,y,error to slice mask image
    '''

    # 1) Read in mask image (created elsewhere from Fineline) - this was done in Well Assignment subdirectory.

    well_mask = io.imread(config['well_mask']['filename'])

    # invert if necessary, by looking at the corner
    if (well_mask[:10,:10]==0).sum() < 0.5:
        well_mask = ~well_mask

    rescale_factor = config['well_mask']['pixel_size']/config['image']['pixel_size']

    # 2) Resize: The scaling for this image is config['well_mask']['pixel_size']

    well_mask = rescale(well_mask,rescale_factor)==1 # convert to binary

    # 3) Slice mask
    mask_xy = matchmask.slice_mask(config,rotation_theta)

    return well_mask, mask_xy

def fit_droplets_to_mask(config,droplets,rotation_theta):
    ''' Fit the droplets DataFrame to the well_mask to determine which droplets are in the same well.

    Inputs:
        - config, the config dictionary created from config
        - droplets, the droplets dataframe (must have rotated coordinates, RX and RY)
        - rotation_theta, the angle of rotation calculated in apply_rotation (called from initialize_droplets)

    Outputs:
        - droplets, the droplets dataframe with well IDs
    '''

    well_mask, mask_xy = initialize_well_mask(config,rotation_theta)

    # Determine image list
    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names']['premerge'])

    droplets = droplets.copy()
    droplets['Well_ID'] = 0
    droplets['Edge'] = False
    removed = []

    for xy in image_idx:
        x = xy[0]
        y = xy[1]

        print 'Fitting droplets to well mask in:',x,y

        # Try to load mask; continue otherwise
        mask = well_mask[mask_xy(x,y,100)]

        if 0 in mask.shape:
            continue

        # Pull out coordinates
        d_idx = (droplets['IndexX']==x) & (droplets['IndexY']==y)
        pos = droplets[d_idx][['RX','RY']].values

        if 0 in pos.shape:
            continue

        # Synthesize image for each set of droplet positions
        syn_img, shifted_pos = matchmask.synthesize_image(pos)

        # Pad images to the same size
        imgs_,pad_shifts = matchmask.pad_images_to_same((syn_img,mask))
        syn_img, mask = imgs_

        shifted_pos[:,0] = shifted_pos[:,0]+pad_shifts[2]
        shifted_pos[:,1] = shifted_pos[:,1]+pad_shifts[0]

        # Translate
        t_shift, t_error, t_phasediff = matchmask.register_translation(syn_img,mask)
        print 'Shift: ',t_shift

        # Updated shifted positions
        shifted_pos[:,1] = shifted_pos[:,1]-t_shift[0]
        shifted_pos[:,0] = shifted_pos[:,0]-t_shift[1]

        # Convert mask to label image
        label_img = matchmask.label_bw(matchmask.binary_fill_holes(mask))

        # Identify pixels in label image correponding to each droplet
        shifted_pos = np.round(shifted_pos).astype('int')

        rmv_idx = ((shifted_pos<0).sum(axis=1)>0) | (shifted_pos[:,0]>=mask.shape[1]) | (shifted_pos[:,1]>=mask.shape[0])
        shifted_pos[rmv_idx,:] = 0

        well_ids = label_img[shifted_pos[:,1],shifted_pos[:,0]]

        well_ids[rmv_idx]=0
        removed.append(rmv_idx.sum())

        # Identify wells on edges
        emask = matchmask.edge_mask(shifted_pos,mask)
        edge_wells = label_img[emask==1]
        on_edge = dict([(a,a in edge_wells) for a in well_ids])
        on_edge[0]=False

        # Store labels from shifted_pos in droplets dataframe
        droplets.loc[d_idx,'Well_ID']=well_ids
        droplets.loc[d_idx,'Edge'] = np.asarray([on_edge[a] for a in well_ids])

    droplets = droplets[droplets['Well_ID']!=0]
    droplets = droplets.reset_index(drop=True)

    # Hash ImageX, ImageY, and Well_ID to create unique wells across whole chip
    droplets['Hash']=droplets[['IndexX','IndexY','Well_ID']].apply(lambda x: hash(tuple(x)),axis=1)

    return droplets

######################################## CLUSTER  #########################

def identify_clusters(config, droplets,show=0,ax=None):
    ''' Identify clusters and assign all droplets to a cluster. Then map apriori labels to clusters.
    Inputs:
        - config, config dictionary
        - droplets, the droplets DataFrame
        - (optional), show (1 to show output, 0 default)
        - (optional), ax (axes handle to plot to, default to None)
    Outputs:
        - droplets, the droplets DataFrame with Cluster column added
        - centroids, n x 2 array of the centroids of clusters found

    '''
    droplets = droplets.copy()

    # project into 2d
    offset=config['barcodes']['cluster']['offset']
    on_plane = cluster.to_2d(cluster.to_simplex(cluster.normalize_vector(droplets[['R','G','B']].values,offset=offset)))

    droplets['PlaneX'] = on_plane[:,0]
    droplets['PlaneY'] = on_plane[:,1]

    # Import parameters from config
    points_to_cluster=config['barcodes']['cluster']['points_to_cluster']
    eps=config['barcodes']['cluster']['eps']
    min_samples=config['barcodes']['cluster']['min_samples']

    # Find cluster centroids and assign all droplets to clusters.
    centroids, labels = cluster.identify_clusters(on_plane, points_to_cluster=points_to_cluster, eps=eps,min_samples=min_samples,show=0)
    droplets['Cluster']=labels

    if show:
        if ax is None:
            fig, ax = plt.subplots()

        for a in droplets['Cluster'].unique():
            idx = droplets['Cluster'].values==a
            ax.plot(on_plane[idx,0],on_plane[idx,1],'.',alpha=0.01)

    return droplets

def map_labels_to_clusters(config, droplets, show=0, ax=None):

    droplets = droplets.copy()

    # Map labels to clusters

    # Read in apriori barcodes
    apriori = {}
    apriori['map'] = kchip_io.read_excel_barcodes(config)
    apriori['barcodes'] = np.vstack([np.asarray(apriori['map'][a]) for a in apriori['map'].keys()])

    # Rearrange 647 and 594
    apriori['barcodes'] = apriori['barcodes'][:,[2, 1, 0]]

    # Project to 2D
    apriori['barcodes_2d']  = cluster.to_2d(cluster.to_simplex(cluster.normalize_vector(apriori['barcodes'])))

    # Map apriori barcodes to clusters
    centroids = droplets.groupby('Cluster')[['PlaneX','PlaneY']].median().values
    assignments, map_, unassigned  = cluster.map_barcodes_to_clusters(apriori['barcodes_2d'],centroids,show=show,ax=ax)

    # Add labels to droplets dataframe
    cluster_id_to_label = dict()
    for i,j in assignments:
        cluster_id_to_label[j] = apriori['map'].keys()[i]
        
    cluster_id_to_barcode = dict()
    for i,j in assignments:
        cluster_id_to_barcode[j] = apriori['map'].values()[i]

    droplets['Label']=[cluster_id_to_label[i] for i in droplets['Cluster']]
    droplets['Barcode']=[cluster_id_to_barcode[i] for i in droplets['Cluster']]
    droplets['Actual_x']=on_plane[:,0]
    droplets['Actual_y']=on_plane[:,1]

    if show:
        d = droplets.groupby('Label').median()[['PlaneX','PlaneY']]
        for label in d.index.values:
            ax.text(d.loc[label,'PlaneX'],d.loc[label,'PlaneY'],label)

<<<<<<< HEAD
    return droplets, on_plane, centroids, cluster_id_to_label, apriori, labels
=======
    return droplets
>>>>>>> interactive

######################################## REGISTRATION  #########################

def initialize_post_wells(config,timepoint):
    ''' Identify wells in post-merge timepoint. Returns post_wells dataFrame.
    Inputs:
        - config, the config dictionary
        - timepoint, the name of the timepoint to analyze (e.g. t0, t1, t2 ...) (should correspond to name in config file)
    Outputs:
        - post_wells, dataFrame'''

    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names'][timepoint])

    post_wells = []
    for xy in image_idx:
        print 'Now analyzing: '+str(xy[0])+','+str(xy[1])

        # Read in image
        post_img = kchip_io.read(config, x=xy[0],y=xy[1],t=timepoint)

        post_wells_ = drop.post_img_to_wells(config,post_img)
        post_wells_.insert(0,'IndexY',xy[1])
        post_wells_.insert(0,'IndexX',xy[0])
        post_wells.append(post_wells_)

    post_wells = pd.concat(post_wells)
    post_wells.reset_index(inplace=True,drop=True)

    return post_wells

def map_pre_to_post(config,timepoint,pre):
    ''' Map the pre-merge wells to wells detected in post-merge images.
    Inputs:
        - config, the config dictionary
        - timepoint, the name of the timepoint to analyze (e.g. t0, t1, t2 ...) (should correspond to name in config file)
        - pre, the wells dataFrame (computed from droplets dataframe by grouping droplets together by their hash)
    Outputs:
         - pre_post_merge, the dataFrame that connects the pre dataframe with wells detected in postmerge images
    '''
    # copy so don't change the dataframe
    pre = pre.copy()

    # Find the post-merge wells
    post = initialize_post_wells(config,timepoint)

    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names']['premerge'])

    # Compute the translation between the corner images
    corners=[(2,2),(2,-2),(-2,2),(-2,-2)]

    xRange = np.unique(image_idx[:,0])
    yRange = np.unique(image_idx[:,1])

    corner_idx = [(xRange[item[0]],yRange[item[1]]) for item in corners]

    translation_ = np.zeros((4,2))
    for i,c in enumerate(corner_idx):
        translation_[i,:]=reg.register(config,c,t='premerge',t2=timepoint)

    translation = translation_.mean(axis=0)

    # Convert everything to global coodinates
    pre = reg.global_coordinates(config,pre)

    post = reg.global_coordinates(config,post)

    # Apply translation to register images
    # # flip coordinates (since x = columns = axis 1, y = rows = axis 0), and transpose
    shift = translation[::-1]

    # # assign wells
    well_assignments, removed_wells = reg.resolve_conflicts(reg.assign_wells(pre,post,shift=shift))

    # Map to post-merge data
    pre = pre.loc[well_assignments[:,0]].reset_index()
    post = post.loc[well_assignments[:,1]].reset_index()

    pre_post_merge = pre.merge(post,left_index=True,right_index=True)

    pre_post_merge.columns = ['Pre_'+a for a in pre.columns]+['Post_'+a for a in post.columns]

    pre_post_merge['Hash']=pre_post_merge[['Pre_IndexX','Pre_IndexY','Pre_Well_ID']].apply(lambda x: hash(tuple(x)),axis=1)

    return pre_post_merge

####################### CONSOLIDATE DROPLETS AND WELLS DATAFRAME ################
def condense_output(droplets,wells):
    ''' Condense output to yield just the wells, droplet constituents, and area + intensity.
    Inputs:
        - droplets, the droplets Dataframe
        - wells, the wells dataframe
    Outputs:
        - condensed, the condensed output dataframe
    '''

    condensed = wells[['Post_Area','Post_Intensity','Hash']]

    condensed.columns = ['Area','Intensity','Hash']

    counted_droplets = droplets.groupby(['Hash','Label'])['IndexX'].count().unstack(level=-1).reset_index()
    condensed = condensed.merge(counted_droplets,on='Hash')
    condensed = condensed.fillna(0.)

    labels = droplets['Label'].unique()
    condensed['Total']=condensed[labels].values.sum(axis=1)
    return condensed

def stack_timepoints(droplets,pre_post_list,timepoint_labels):
    ''' Create a condensed output with stacked timepoints.
    Inputs:
        - droplets, the droplets DataFrame
        - pre_post_list, a list of the wells dataFrames from each timepoints
        - timepoint_labels, a list of labels for each timepoint, e.g. [t0, t1, t2, ...]
    Outputs:
        - condensed, the condensed output DataFrame
    '''

    pruned_list = []
    for df in pre_post_list:
        pruned_list.append(df[['Hash','Post_Area','Post_Intensity',]].set_index('Hash'))

    stacked = pd.concat(pruned_list,axis=1)

    column_labels = reduce(lambda x,y: x+y, [[t + '_Area',t] for t in timepoint_labels])
    stacked.columns = column_labels

    condensed = condense_output(droplets,pre_post_list[0])
    condensed_columns = [c for c in condensed.columns if c not in ['Area','Intensity']]
    condensed = condensed[condensed_columns]

    out = condensed.set_index('Hash').merge(stacked,left_index=True,right_index=True).reset_index()

    return out
