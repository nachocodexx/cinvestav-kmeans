import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.colors as mcolors
import random
from utils import normalize
style.use('ggplot')
COLORS = list(mcolors.CSS4_COLORS.keys())


class KMeans(object):
    def __init__(self,*args,**kwargs):
        self.df = kwargs.get("df")
        self.rowsLen , self.columnsLen= self.df.shape
        self.k =kwargs.get("k",1)
        self.centers = np.random.uniform(0,.8,self.k*self.columnsLen).reshape(self.k,self.columnsLen)
        self.plot_title="K-means"
        self.debug=kwargs.get("DEBUG",True)
        generateColors = np.vectorize(lambda x : random.choice(COLORS))
        self.colors = generateColors(np.random.uniform(0,1,self.k))

    def updateCenters(self,labels):
        centroids = np.zeros(self.k*self.columnsLen).reshape(self.k,self.columnsLen)
        for i in range(self.k):
            data= self.df[labels==i]
            mean=data.mean(axis=0)
            centroids[i]=mean
        return centroids
            
    def fit(self,**kwargs):
        max_iterations = kwargs.get("epochs")
        labels=np.zeros(self.rowsLen)
        # self.plot(0)
        inertia=0
        for i in range(max_iterations):
            for index,x in enumerate(self.df.to_numpy()):
                distancesToCenters=np.array(list(map(lambda y: self.__distancesBetweenPoints(x,y) , self.centers)))
                minDistanceIndex = distancesToCenters.argmin()
                minDistance = distancesToCenters.min()
                labels[index]=minDistanceIndex
            self.centers=self.updateCenters(labels)
            # self.plot(i)
        self.inertia = np.zeros(self.k)
        for i in range(self.k):
            dataPoints = self.df[labels==i].to_numpy()
            inertia=np.zeros(dataPoints.shape[0])
            for index,x in enumerate(dataPoints):
                distances= np.square(self.__distancesBetweenPoints(x,self.centers[i]))
                inertia[index]=distances
            self.inertia[i]=inertia.sum()
        
        self.inertia = self.inertia.sum()

    def __distancesBetweenPoints(self,x,y):
        if not (x.shape == y.shape):
            raise Exception("The points have diff shape")
        res = np.square(x-y)
        return np.sum(res)
    def plot(self,index):
        _,ax = plt.subplots()
        for index,color in enumerate(self.colors):
            pointX = self.centers[index,0]
            pointY = self.centers[index,1]
            ax.scatter(pointX,pointY,zorder=1,c=color,s=300,label=color,marker="*")
        # xCenter = self.centers[:,0]
        # yCenter = self.centers[:,1]
        # ax.scatter(xCenter,yCenter,100,zorder=1,color=self.colors,marker="*")
        # ax.scatter(xCenter,yCenter,100,zorder=1,color=["r","b","g"],marker="*")
        df.plot.scatter(x=0,y=1,zorder=0,ax=ax,color="black",alpha=.3)
        ax.set_title(self.plot_title)
        ax.legend()
        plt.savefig("images/frame_{}".format(index+1))
        # plt.show()



        pass
if __name__ == "__main__":
    df  = pd.read_csv("data/01.csv", header=None)
    df.drop(df.columns[[2]],axis=1,inplace=True)
    #_____________WINES____________________________ 
    # df  = pd.read_csv("data/wines.csv")
    # df =normalize(df) 
    # df=df.drop(["Customer_Segment"],axis=1)
    #__________________________________________
    epochs = 10
    tests=[]
    K=range(1,10)
    inertias=[]
    for k in K:
        kmeans =KMeans(df=df,k=k)
        print("k = {}".format(k))
        kmeans.fit(epochs=epochs)
        inertias.append(kmeans.inertia)
        print("_______________________________")
    plt.plot(K,inertias,'b*-')
    plt.show()
    # kmeans.plot()

