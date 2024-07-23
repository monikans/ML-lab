# PCA and LDA 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def plot_graph(X_projected,y,xlabel,ylabel):
  plt.scatter(X_projected[:,0],X_projected[:,1], c=y, cmap = 'jet')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()
X,y = load_iris(return_X_y = True)
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
n_components = 2

pca = PCA(n_components= n_components)
x_pca = pca.fit_transform(X_scaler)
print(f'the shape of transformed data is: {x_pca.shape}')
print(f'the transformed data is: {x_pca}')
plot_graph(x_pca,y,'pca1','pca2')

lda = LinearDiscriminantAnalysis(n_components= n_components)
x_lda = lda.fit_transform(X_scaler,y)
print(f'the shape of transformed data is: {x_lda.shape}')
print(f'the transformed data is: {x_lda}')
plot_graph(x_lda,y,'pca1','pca2')
