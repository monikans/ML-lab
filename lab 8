# K-means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
df = pd.read_csv("/content/Iris.csv")
df.head()

def kmeans(X,K,max_iters):
  centroids = X[:K]
  for _ in range(max_iters):
    labels = np.argmin(np.linalg.norm(X[:,np.newaxis]-centroids ,axis = 2),axis= 1)
    new_centroids = np.array([X[labels==k].mean(axis=0) for k in range( K)])
    if np.all(centroids==new_centroids):
      break
    centroids = new_centroids
  return labels,centroids
X=np.array(df.iloc[:,:-1].values)
labels,c=kmeans(X,3,200)
print(labels)
print(c)
plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(c[:,0],c[:,1],marker = "x",color = "red")
plt.show()
