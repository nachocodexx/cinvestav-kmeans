#
from time import time
#
from sklearn.datasets.samples_generator import make_blobs
#Numpy library : Multi-dimensional arrays and matrices
import numpy as np
# Pandas library : Data manipulation and analysis
import pandas as pd
# Matplotlib library for plotting or data visualization
import matplotlib.pyplot as plt
# Import the styles : Change the style of the plot
from matplotlib import style
# Import the colors
import matplotlib.colors as mcolors
# Random 
import random
# Utils functions
from utils import normalize
# Using the ggplot style
style.use('ggplot')
# List of colors from matplotlib
COLORS = list(mcolors.CSS4_COLORS.keys())


class KMeans(object):
    """
        Clustering algorithm KMeans that receive by argument the number of centroids and the dataset
    """
    def __init__(self,*args,**kwargs):
        # Getting the dataset from kwargs
        self.df = kwargs.get("df")
        # Getting the k from kwargs
        self.k =kwargs.get("k",1)
        # Getting the number of rows from dataframe shape attribute
        self.rows_length , self.columns_length= self.df.shape
        # Initilize centers randomly
        self.centers = self.initilizeCenters()
        # Plot's title
        self.plot_title="K-means"
        # DEBUG mode by default is True
        self.DEBUG=kwargs.get("DEBUG",False)
        self.ELBOW=kwargs.get("ELBOW",False)
        # Generating colors randomly function for the centroids
        generateColors = np.vectorize(lambda x : random.choice(COLORS))
        #  Centroid's colors depending of the k 
        self.colors = generateColors(np.random.uniform(0,1,self.k))
        # self.colors = ["red","blue","green"]

    """Initilize centers randomly method"""
    def initilizeCenters(self):
        return np.random.uniform(0,1,self.k*self.columns_length).reshape(self.k,self.columns_length)
    """
        Update centers method
        (Labels:Vector[Float]) => Matrix(K,columns_length)
    """
    """Update centers method
        Description: Get all the datapoint using the labels and get the mean and assign to the centroid
    """
    def updateCenters(self,labels):
        centroids = np.zeros(self.k*self.columns_length).reshape(self.k,self.columns_length)
        for i in range(self.k):
            data= self.df[labels==i]
            mean=data.mean(axis=0)
            centroids[i]=mean
        return centroids
            
    """
    Fit method, using the dataset for training the model
    (epochs:Int)=>Unit
    """
    def fit(self,**kwargs):
        # Number of epochs or iterations 
        #O(1)
        max_iterations = kwargs.get("epochs")
        # Labels vector
        #O(n)
        labels=np.zeros(self.rows_length)
        # self.plot(0)
        # Loop through max_iterations
        ##############################################################

        #> O(max_iterations * rows * k )
        for i in range(max_iterations):
            # Loop through the number of datapoints
            for index,x in enumerate(self.df.to_numpy()):
                # Calculating the Euclidean distances from the datapoint to all the k centers
                distancesToCenters=np.array(list(map(lambda y: self.__distancesBetweenPoints(x,y) , self.centers)))
                # Getting the index of the min center
                minDistanceIndex = distancesToCenters.argmin()
                #minDistance = distancesToCenters.min()
                # Saving the index of the closest center 
                labels[index]=minDistanceIndex
            # Update the center 
            self.centers=self.updateCenters(labels)
        #############################################################

            # Plotting the result only if DEBUG is True
            # if(self.DEBUG):
                # self.plot(i,labels)
            print("[{0}] Calculating new centroids...".format(i))
        # Calculating the inertia only if ELBOW is True
        self.plot(0,labels)
        if(self.ELBOW):
            self.inertia= self.calculateInertia(labels)
    """Calculating the inertia method
    The inertia is the sum of the squared distances of all the cluster data points and the cluster's centroid
    """
    def calculateInertia(self,labels):
        # Creating a total vector with shape (k,)
        total = np.zeros(self.k)
        # Loop through 0 to k
        for i in range(self.k):
            # Getting the data points inside the ith cluster
            dataPoints = self.df[labels==i].to_numpy()
            allDistances=np.zeros(dataPoints.shape[0])
            # Loop through datapoints
            for index,x in enumerate(dataPoints):
                # Calculating the distances from datapoint x to the centroid of that cluster and square it
                allDistances[index]=np.square(self.__distancesBetweenPoints(x,self.centers[i]))
            # Sum all them up
            total[i]=allDistances.sum()
        # Returning the total 
        return total.sum()

        


    """Euclidean distances """
    def __distancesBetweenPoints(self,x,y):
        # Check if the points have the same shape
        if not (x.shape == y.shape):
            raise Exception("The points have diff shape")
        # Square it the difference between points
        res = np.square(x-y)
        # Sum the differences and sqrt
        return np.sqrt(np.sum(res))


    def plot(self,index,labels):
        # Creating subplots , we only use Axes object 
        _,ax = plt.subplots()
        # Iterate through the colors, colors is a narray with shape (k,)
        # for index,color in enumerate(self.colors):
            # Getting the x coordinate of the centroid
            # pointX = self.centers[index,0]
            # Getting the y coordinate of the centroid
            # pointY = self.centers[index,1]
            # Plotting using the Axes's scatter method   
        ax.scatter(self.centers[:,0],self.centers[:,1],zorder=1,c="black",s=400,marker="*")
        # print(points)
        # Using the DataFrame's plot method to plot the datapoints and passing by argument the ax object 
        # self.df.plot.scatter(x=0,y=1,zorder=0,ax=ax,color="red",alpha=.3)
        for index,x in enumerate(self.df.to_numpy()):
            ax.scatter(x[0],x[1],c=self.colors[int(labels[index])],zorder=0)
        # ax.scatter(self.df)
            
        # Axes's set_title method to show a title
        ax.set_title(self.plot_title)
        # Axes's legend method, only to show informatioabout the centroid(e.g the color )
        # ax.legend()
        # Saving the current plot as image
        # plt.savefig("images/frame_{}".format(index+1))
        # plt.show()

        # xCenter = self.centers[:,0]
        # yCenter = self.centers[:,1]
        # ax.scatter(xCenter,yCenter,100,zorder=1,color=self.colors,marker="*")
        # ax.scatter(xCenter,yCenter,100,zorder=1,color=["r","b","g"],marker="*")
        # plt.show()



        pass
if __name__ == "__main__":
    start_time = time()
    # df  = pd.read_csv("data/01.csv", header=None)
    # df.drop(df.columns[[2]],axis=1,inplace=True)
    #_____________WINES____________________________ 
    # df  = pd.read_csv("data/wines.csv")
    # df =normalize(df) 
    # df=df.drop(["Customer_Segment"],axis=1)
    #__________________________________________

    ############################
    epochs=100
    k=40
    ###########################
    data , y_true = make_blobs(n_samples = 10000, centers=k, cluster_std=1)
    df =  normalize(pd.DataFrame(data=data))
    df.to_csv("gaussian.csv")
    ##################################3
    kmeans = KMeans(df=df,k=k,DEBUG=True)
    # kmeans.fit(epochs=epochs)
    execution_time = time() - start_time
    print("Execution time: [{}s]".format(execution_time))
    # plt.show()
    ##################
    # epochs = 10
    # tests=[]
    # K=range(1,10)
    # inertias=[]
    # _,ax = plt.subplots()
    # for k in K:
        # kmeans =KMeans(df=df,k=k,ELBOW=True)
        # print("k = {}".format(k))
        # kmeans.fit(epochs=epochs)
        # inertias.append(kmeans.inertia)
        # print("_______________________________")
    # ax.set_title("Elbow method")
    # ax.set_xlabel("K")
    # ax.set_ylabel("Inertia")
    # ax.plot(K,inertias,'ro-')
    # plt.show()
    # kmeans.plot()

