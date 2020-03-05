# Basic utilities
import numpy as np
import copy

# sklearn
from sklearn.cluster import dbscan
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression

# scipy
from scipy.spatial.distance import cdist

# Plotting
import matplotlib.pyplot as plt

def normalize_vector(vectors,offset=0):
    '''Normalize vectors to magnitude = 1, with possibility to subtract offset first
    Inputs:
    - vectors: (k row x n column array) of k n-dimensional vectors
    - offset: (1 row x n column array) of static offset to apply (default=0)
    Outputs:
    - normalized: (k row x n column array) of normalized vectors, norm = 1
    '''
    magnitudes = np.sqrt(np.sum((vectors-offset)**2,axis=1))
    return (vectors-offset)/magnitudes.reshape(len(magnitudes),1)

def to_simplex(vectors,offset=0):
    '''Project vectors onto simplex plane, with possibility to subtract offset first
    Inputs:
    - vectors: (k row x n column array) of k n-dimensional vectors
    - offset: (1 row x n column array) of static offset to apply (default=0)
    Outputs:
    - normalized: (k row x n column array) of vectors projected on simplex
    '''
    vectors_ = vectors-offset
    return vectors_/np.sum(vectors_,axis=1).reshape((vectors_.shape[0],1))

def to_2d(vectors,basis = np.asarray([[-1., -1.],[1., -1.],[0., 1.]])):
    '''Project vectors from 3d onto 2d plane using supplied basis vectors
    Inputs:
        - vectors (k row x 3 column np array) of k 3d vectors
        - (optional) basis (3 row x 2 column np array) of basis vectors for projection
            defaults to basis vectors of 1,1,1 simplex plane
    '''
    return np.dot(vectors,basis)

def compute_centroids(positions,labels,show=0):
    '''Compute centroids of the supplied clusters
    Inputs
    - positions (n x 2 array): a list of all points, 2 dimensional
    - labels (n x 1 array, k unique integers): labels for points for which of the k clusters they belong to
    Outputs
    - centroids (k x 2 array): a list centroids for each of the k clusters
    '''
    class_labels = np.unique(labels)

    centroids = np.zeros((len(class_labels),2))
    for i,l in enumerate(class_labels):
        centroids[i,:]=np.mean(positions[labels==l,:],axis=0)

    if show:
        plt.scatter(centroids[:,0],centroids[:,1],s=10,c='r')

    return centroids

def identify_clusters(positions, points_to_cluster=2000, eps=0.025, min_samples=6,seed=0,show=0):
    '''Use dbscan to identify the initial cluster positions, in 2D plane
    Inputs
    - Positions (n x 2 array): a list of all points to cluster, 2 dimensional vectors in plane
    - points_to_cluster (integer): the number of points to randomly sample to cluster using dbscan
    - eps: dbscan parameter eps, the max distance between points to co-cluster
    - min_samples: dbscan parameter min_samples, the min number of points in a cluster
    - seed (int): seed for random number generator for sampling

    Outputs
    - labels (n x 1 array, integers): cluster ids for each point, noise: label=-1
    - centroids (n x 2 array): centroids of clusters (except for noise point)
    '''
    sz = positions.shape[0]

    # Seed random generator for sampling points
    np.random.seed(seed=seed)
    choose_points = np.random.choice(range(sz),size=(np.min([points_to_cluster,sz]),))
    pos_ = positions[choose_points,:]

    core, labels = dbscan(pos_,eps=eps,min_samples=min_samples)

    if show:
        # addplot.cluster_plot(pos_,labels)
        print ('Selected '+ str(points_to_cluster)+ ' points, with random seed at: ' + str(seed))
        print ('Removed '+ str(float(np.sum([labels==-1]))/len(labels)*100)+ '% of Points')
        print ('Found ' +str(len(np.unique(labels))-1)+ ' Clusters')

    # Compute centroids, of all clusters but noise clusters (label=-1)
    centroids = compute_centroids(pos_[labels!=-1,:],labels[labels!=-1])

    # Assign all points to cluster centroids
    assignments = assign_cluster(positions,centroids,show=show)

    return centroids, assignments

def assign_cluster(points, centroids,show=0):
    distances = cdist(points, centroids)
    assignments = np.argmin(distances,axis=1)

    if show:
        addplot.cluster_plot(points,assignments)
        ax = plt.gca()
        ax.scatter(centroids[:,0],centroids[:,1],s=10,c='k',edgecolors='w')

    return assignments

def munkres(b1,b2,show=0,ax=None):
    '''Returns assignments from one barcode set to a second using Munkres (Hungarian) algorithm
    Inputs:
    - b1: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    - b2: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    - (optional) show: [0, 1]
    Outputs
    - assignments (n x 2 numpy array, integers) of indices in b1 to indices in b2
    '''

    assignments = linear_sum_assignment(cdist(b1, b2))
    assignments = np.transpose(np.asarray(assignments))

    if show:
        plot_shift(b1[assignments[:,0],:],b2[assignments[:,1],:],ax=ax)

    return assignments

def map_munkres(b1,b2,show=0):
    '''Returns a linear map from set 1 to set 2
    Inputs:
    - b1: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    - b2: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    - (optional) show: [0, 1]
    Returns:
    - model: a sklearn model, use model.predict([nx2]) to map
    - b1_pred: the mapped coordinates
    '''

    model = LinearRegression()
    model.fit(b1,b2)

    b1_pred = model.predict(b1)
    if show:
        plot_shift(b2,b1_pred)

    return model, b1_pred

def map_barcodes_to_clusters(barcodes,clusters,show=0,ax=None):
    ''' '''
    # Make initial assignments
    assignments_ = munkres(barcodes,clusters)

    # Compute linear map
    map_, pred_ = map_munkres(barcodes[assignments_[:,0],:],clusters[assignments_[:,1],:])

    # Redo assignments, now with proper map
    assignments = munkres(map_.predict(barcodes),clusters,show=show,ax=ax)

    # Check for unassigned
    if len(assignments) < len(barcodes):
        unassigned = [i for i in np.arange(len(barcodes)) if i not in assignments[:,0]]
    else:
        unassigned = []

    return assignments, map_, unassigned

def plot_shift(b1,b2,ax=None):
    '''Plots map from one set of coordinates to another set of coordinates
    Inputs:
    - b1: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    - b2: (n x 2 numpy array) set of barcodes (in 2 dimensions)
    '''
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.vstack((b1[:,0],b2[:,0])),np.vstack((b1[:,1],b2[:,1])))

class ReactiveCluster(object):
    ''' Interactive clustering for manual correction of initial assingments by identify_clusters.
    Inputs:
        - wells, a dataFrame as loaded from wells.xlsx
    Interaction:
        - Left click: Add cluster centroid at mouse location
        - Right click: Remove cluster centroid closest to mouse location, and update plot
    Returns:
        - Instance of ReactiveCluster Class, with following accessible attributes:
            - centroids, an (n x 2) array of cluster centroid locations
            - labels, (m x 1) assignments for each point
            - points, (m x 2) array of points
    '''

    def __init__(self, config, wells):

        # Import parameters from config
        offset=config['barcodes']['cluster']['offset']
        points_to_cluster=config['barcodes']['cluster']['points_to_cluster']
        eps=config['barcodes']['cluster']['eps']
        min_samples=config['barcodes']['cluster']['min_samples']

        # Initialize clustering
        self.droplets = wells.copy(deep=True)

        # Initialize barcodes dictionary
        barcodes = dict()

        # Compile droplets colors into np array
        droplets_colors = self.droplets[['R','G','B']].values
        # Subtract offsets and project onto plane
        self.points = to_2d(to_simplex(normalize_vector(droplets_colors,offset=offset)))

        # Use DBSCAN algorithm to estimate cluster centroids
        centroids, labels = identify_clusters(self.points, points_to_cluster=points_to_cluster, eps=eps,min_samples=min_samples,show=0)
        self.droplets['Cluster']=labels
        self.centroids = centroids

        #Initialize plot
        self.fig, self.ax = plt.subplots(1, 1)
        self.cid = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.draw()

    def draw(self):
        for cluster_id in self.droplets['Cluster'].unique():
            idx = self.droplets['Cluster'].values==cluster_id
            self.ax.plot(self.points[idx,0],self.points[idx,1],'.',alpha=0.01,picker=5)
            self.ax.text(self.points[idx,0].mean(),self.points[idx,1].mean(),cluster_id)

        self.ax.plot(self.centroids[:,0],self.centroids[:,1],'rx')
        self.ax.set_title('Number of centroids:' + str(self.centroids.shape[0]))
        plt.draw()

    def update(self):
        self.ax.clear()
        self.droplets['Cluster'] = assign_cluster(self.points, self.centroids)
        self.draw()

    def on_pick(self, event):
        print ('detected pick')
        try:
            pos = np.asarray([(event.mouseevent.xdata,event.mouseevent.ydata)])

            if event.mouseevent.button == 1:
                # add centroid
                self.centroids = np.vstack([self.centroids,pos])
            else:
                # remove centroid
                distances = cdist(self.centroids, pos)
                select = np.argmin(distances)
                self.centroids = np.delete(self.centroids,select,axis=0)

            self.update()
        except Exception as e:
            print (e.message)

    def output(self):
        self.droplets['Cluster'] = assign_cluster(self.points, self.centroids)
        return self.droplets, self.centroids
