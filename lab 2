# Contour plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
data = np.random.rand(100,3)
df = pd.DataFrame(data,columns=['Feature 1','Feature 2','Feature 3'])
plt.figure(figsize=(12,8))
x,y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
z = np.sin(x*np.pi)*np.cos(y*np.pi)
contour = plt.contour(x,y,z,20,cmap = 'viridis')
plt.colorbar(contour)
plt.tight_layout()
plt.show()

# A star
def h(n):
  H = {'A':3,'B':4,'C':6,'D':2,'G':0,'S':5}
  return H[n]

def a_star(graph,start,goal):
  ol = [start]
  cl = set()
  g = {start:0}
  parent = {start:start}
  while ol:
    ol.sort(key = lambda v: g[v]+h(v),reverse = True)
    n = ol.pop()
    if n==goal:
      reco_path = []
      while parent[n]!=n:
        reco_path.append(n)
        n = parent[n]
      reco_path.append(start)
      reco_path.reverse()
      print(f'path found: {reco_path}')
      return reco_path
    for (m,wt) in graph[n]:
      if m not in ol and m not in cl:
        ol.append(m)
        parent[m] = n
        g[m] = g[n]+wt
      else:
        if g[m]>g[n]+wt:
          g[m] = g[n]+wt
          parent[m] = n
          if m in cl:
            cl.remove(m)
            ol.append(m)
    cl.add(n)
  print("path doesnt exist")
  return None
graph = {
    'S': [('A', 1), ('G', 10)],
 'A': [('B', 2), ('C', 1)],
 'B': [('D', 5)],
 'C': [('D', 3),('G', 4)],
 'D': [('G', 2)]
}
a_star(graph,'S','G')
