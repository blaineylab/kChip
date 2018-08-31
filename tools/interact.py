import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bqplot.pyplot as bqplt #for interactive plotting
from scipy.spatial.distance import cdist
from bqplot import colorschemes
from itertools import cycle

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
def get_image(config,row,timepoint='t2'):
    return read(config,int(row['IndexX']),int(row['IndexY']),timepoint)[:,:,config['image']['bugs']]

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

def slice_well(config,row,timepoint='t2',size=25):
    return get_image(config,row,timepoint=timepoint)[bbox(config,row['ImageX'],row['ImageY'],size=25)]

def montage_by(config,df,timepoint='t2',n=10,size=25,return_index=False,**kwargs):
    clip_edge = int((float(size)/2))+1
    b = config['image']['size']-1
    s = select(select_range(df,ImageX=[clip_edge,b-clip_edge],ImageY=[clip_edge,b-clip_edge]),**kwargs).sample(n)
    arr = np.dstack([slice_well(config,r,timepoint=timepoint,size=size) for i,r in s.iterrows()]).transpose([2,0,1])

    if not return_index:
        return montage.montage2d(arr)
    else:
        return montage.montage2d(arr), s.index

###### INTERACTING WITH PLOTS ########

class InteractiveCluster(object):
    ''' Create an 2D scatterplot such that clusters can be selected by point and clickself. (Requires bqplot library)
    For best performance, don't cluster more than a few thousand points.

    There are no outputs, but you can get the cluster centroids from the "centroids" property and labels from the "labels" property.

    Input:
    - points (n x 2 numpy array of the points to cluster)
    - centroids (m x 2 numpy array of initial cluster centroids)
    '''

    def __init__(self, points, centroids):
        self.figure = bqplt.figure(title='Click on point to add cluster / Click on centroid to remove a cluster')
        self.points = points
        self.centroids = centroids

        self.sca_points = self.plot(self.points,style='Points')
        self.sca_points.on_element_click(self.add_cluster)

        self.sca_centroids = self.plot(self.centroids,style='Centroids')

        self.sca_centroids.on_element_click(self.remove_cluster)

        self.cluster()
        self.drag_flag = 0

    def plot(self,arr,style):
        if arr.shape[0]:
            sca = bqplt.scatter(arr[:,0],arr[:,1])
        else:
            sca = bqplt.scatter([1],[1])
            sca.x = sca.y = np.asarray([])

        if style == 'Points':
            sca.default_size = 5
            sca.default_opacities = [0.05]

        else:
            sca.default_size = 50
            sca.default_opacities = [1]
            sca.stroke = 'black'
            sca.colors = self.make_color_set(arr.shape[0])

        return sca

    def cluster(self,points=None):
        if points is None:
            distances = cdist(self.points, self.centroids)
            self.labels = np.argmin(distances,axis=1)
            self.sca_points.colors = self.return_colors()
        else:
            distances = cdist(points, self.centroids)
            labels = np.argmin(distances,axis=1)
            return labels

    def make_color_set(self, n):
        A = np.arange(n)
        B = colorschemes.CATEGORY20
        return zip(*zip(A, cycle(B)))[1]

    def return_colors(self):
        colors = np.asarray(self.make_color_set(np.max(self.labels)+1))
        return list(colors[self.labels.astype('int')])

    def add_cluster(self,obj,target):
        selected = np.asarray([target['data']['x'],target['data']['y']])

        if self.centroids.shape[0]:
            self.centroids = np.vstack((self.centroids,selected))
        else:
            self.centroids = selected

        self.update()

    def remove_cluster(self,obj,target):

        selected = np.asarray([target['data']['x'],target['data']['y']])

        # Find point
        where = np.where(np.sum(self.centroids==selected,1)==2)[0][0]
        self.centroids = np.delete(self.centroids,where,axis=0)

        self.update()

    def update(self):
        self.update_centroids()
        self.cluster()
        self.figure.title = str(self.centroids.shape[0])+' clusters currently identified.'

    def update_centroids(self):
        self.sca_centroids.x = self.centroids[:,0]
        self.sca_centroids.y = self.centroids[:,1]
        self.sca_centroids.colors = self.make_color_set(self.centroids.shape[0])


class LinkedPlot(object):
    ''' Create a linked input and output plot(histogram or scatterplot), where brush selection of the input plot can be used to highlight points in the output plot.
    Input to construct function:
    - DataFrame (pandas)

    To initialize call the constructor LinkedPlot(df) with your dataframe as "df". Then initialize the Input and Output plots.

    Methods:

        Input (function):
        Initialize the input plot with the supplied column names in the dataframe. A single argument is a histogram, 2 arguments is a scatterplot.
            - x, a column name in dataframe
            - y (optional), a column name in dataframe

        Output (function):
        Initialize the output plot with the supplied column names in the dataframe. A single argument is a histogram, 2 arguments is a scatterplot.
            - x, a column name in dataframe
            - y (optional), a column name in dataframe
    '''


    def __init__(self,data):
        self.data = data
        self.data = self.data.assign(_Index=0)
        self.figures = {'Input':None,'Output':None}

    def parse(self, x=None,y=None):
        if (x is None) & (y is None):
            return None
        elif (y is None):
            return self.Hist(x)
        else:
            return self.Scatter(x,y)

    def Input(self, x, y=None):
        fig = self.parse(x=x,y=y)
        self.setup_brush(fig)

        self.figures['Input'] = fig

    def Output(self, x, y=None):
        fig = self.parse(x,y)

        self.figures['Output'] = fig

    def Hist(self, x):

        sample = self.data[x]

        # plotting
        hist_x = bqplt.LinearScale()
        hist_y = bqplt.LinearScale()

        h = bqplt.Hist(sample=sample, scales={'sample': hist_x, 'count': hist_y})

        h_xax = bqplt.Axis(scale=hist_x, label=x, grids='off', set_ticks=True)
        h_yax = bqplt.Axis(scale=hist_y, label='Counts', orientation='vertical', grid_lines='none')

        # construct figure
        fig = bqplt.Figure(marks=[h], axes=[h_xax, h_yax])

        return fig

    def Scatter(self, x, y, **kwargs):
        sc_x = bqplt.LinearScale()
        sc_y = bqplt.LinearScale()

        s = bqplt.Scatter(x=self.data[x], y=self.data[y],
                        scales={'x': sc_x, 'y': sc_y})

        sc_xax = bqplt.Axis(label=(x), scale=sc_x)
        sc_yax = bqplt.Axis(label=(y), scale=sc_y, orientation='vertical')

        fig = bqplt.Figure(marks=[s], axes=[sc_xax, sc_yax])
        return fig

    def setup_brush(self,fig):

        if type(fig.marks[0]) is bqplt.Scatter:
            self.brush = bqplt.BrushSelector(x_scale=fig.marks[0].scales['x'], y_scale = fig.marks[0].scales['y'], marks=fig.marks, color='red')
            self.brush.observe(self.brush_callback, names=['selected'])
            self.brush.observe(self.brush_callback, names=['brushing'])

        elif type(fig.marks[0]) is bqplt.Hist:
            self.brush = bqplt.BrushIntervalSelector(scale=fig.marks[0].scales['sample'], marks=fig.marks, color = 'red')
            self.brush.observe(self.brush_callback, names=['selected'])
            self.brush.observe(self.brush_callback, names=['brushing'])

        else:
            print 'Type unknown.'

        fig.interaction = self.brush

    def brush_callback(self,change):
        if(not self.brush.brushing):
            if type(self.figures['Input'].marks[0]) is bqplt.Scatter:
                selected = np.asarray(self.brush.selected).T
                self.update_index(**dict(zip([_.label for _ in self.figures['Input'].axes],selected)))

            else:
                x = [_.label for _ in self.figures['Input'].axes if _.label != 'Counts'][0]
                self.update_index(**{x:self.brush.selected})

        if type(self.figures['Output'].marks[0]) is bqplt.Scatter:
            self.update_Scatter()

        else:
            self.update_Hist()


    def update_index(self,**kwargs):
        sets = [self.data.query(item+'>='+str(kwargs[item][0])+' & '+item+'<='+str(kwargs[item][1])).index for item in kwargs]
        intersect = reduce(lambda a,b: a.intersection(b),sets)

        self.data['_Index'] = 0
        self.data.loc[intersect,'_Index'] = 1

    def update_Scatter(self):
        colors = ['blue','red']
        self.figures['Output'].marks[0].colors = [colors[i] for i in self.data['_Index'].values.astype('int')]

    def update_Hist(self):
        scales = self.figures['Output'].marks[0].scales
        x = [_.label for _ in self.figures['Output'].axes if _.label != 'Counts'][0]

        h0 = bqplt.Hist(sample=self.data.query('_Index==0')[x], scales=scales, colors =['blue'])
        h1 = bqplt.Hist(sample=self.data.query('_Index==1')[x], scales=scales, colors = ['red'])

        self.figures['Output'].marks = [h0,h1]
