import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import rank
from skimage.transform import resize

#### FLATFIELD CORRECTION ########

def normby(df, groupby, background):
    df = df.copy()
    df['Norm']=(df['Value']-background)/(df.groupby(groupby).transform('mean')['Value']-background)
    return df

def average_over_bin(df,bins):
    df = df.copy()

    df['xbin'] = pd.cut(df['ImageX'],bins=bins,labels=False)
    df['ybin'] = pd.cut(df['ImageY'],bins=bins,labels=False)

    tf = df.groupby(['xbin','ybin'],as_index=False).mean()[['xbin','ybin','Norm']]

    grid = np.zeros((tf['ybin'].max()+1,tf['xbin'].max()+1))
    grid[tf['ybin'].values.astype('int'),tf['xbin'].values.astype('int')]=tf['Norm']

    return grid

def average_grid(grid):

    t = 65535*(grid-grid.min())/(grid.max()-grid.min())
    grid2 = rank.mean(t.astype('uint16'),np.ones((3,3)))
    grid2_u = grid2*(grid.max()-grid.min())/65535+grid.min()

    return grid2_u

def construct_flatfield(df,bins):
    grid = average_grid(average_over_bin(df,bins))
    flatfield = resize(grid,(1024,1024),mode='symmetric')
    return flatfield

def flatfield_correct(row,flatfield_img,background):
    return (row['Value']-background)/flatfield_img[int(row['ImageY']),int(row['ImageX'])]+background
