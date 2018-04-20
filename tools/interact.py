import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..io import read

from skimage.util import montage

############ INTERACTING WITH DATAFRAMES ############

def select(df,**kwargs):
    idx = df[df.columns[0]]==df[df.columns[0]]
    for item in kwargs.keys():
        idx = idx & (df[item]==kwargs[item])
    return df.loc[idx]

def select_range(df,**kwargs):
    sets = [df.query(item+'>='+str(kwargs[item][0])+' & '+item+'<='+str(kwargs[item][1])).index for item in kwargs]
    intersect = reduce(lambda a,b: a.intersection(b),sets)
    return df.loc[intersect]

###### INTERACTING WITH IMAGES ########
def get_image(config,row):
    return read(config,int(row['IndexX']),int(row['IndexY']),'t2')[:,:,config['image']['bugs']]

def show_image(config,row,ax=None,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(
        get_image(config,row),
        **kwargs
    )

def overlay_well(rows,ax=None):
    ax.plot(rows['ImageX'],rows['ImageY'],color='red',marker='o',markerfacecolor='none',markersize=20,ls='')

def bbox(config,x,y,size=25):
    s = float(size)/2
    select = np.asarray(((-s,-s),(s,s)))+[x,y]
    b = config['image']['size']-1
    box = (bound(bound(select,0),b,less=False)).T.astype('int')
    return [slice(box[1,0],box[1,1]),slice(box[0,0],box[0,1])]

def bound(arr,b,less=True):
    arr = arr
    if less:
        arr[arr<=b] = b
    else:
        arr[arr>=b] = b
    return arr

def slice_well(config,row,size=25):
    return get_image(config,row)[bbox(config,row['ImageX'],row['ImageY'],size=25)]

def montage_by(config,df,n=10,size=25,return_index=False,**kwargs):
    clip_edge = int((float(size)/2))+1
    b = config['image']['size']-1
    s = select(select_range(df,ImageX=[clip_edge,b-clip_edge],ImageY=[clip_edge,b-clip_edge]),**kwargs).sample(n)
    arr = np.dstack([slice_well(config,r,size=size) for i,r in s.iterrows()]).transpose([2,0,1])

    if not return_index:
        return montage.montage2d(arr)
    else:
        return montage.montage2d(arr), s.index
