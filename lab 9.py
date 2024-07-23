#single and complete linkage

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data[:6]
def proximity_matrix(data):
  n = data.shape[0]
  proximity_matrix = np.zeros((n,n))
  for i in range(n):
    for j in range(i+1,n):
      proximity_matrix[i,j] = np.linalg.norm(data[i]-data[j])
      proximity_matrix[j,i] = proximity_matrix[i,j]
  return proximity_matrix

def plot_dendrogram(data,method):
  linkage_matrix = linkage(data,method=method)
  dendrogram(linkage_matrix)
  plt.title(f'dendrogram-{method}linkage')
  plt.xlabel("data points")
  plt.ylabel("distance")
  plt.show()

print("proximity_matrix: ")
print(proximity_matrix(data))
plot_dendrogram(data,'single')
plot_dendrogram(data,'complete')
