#3d plot
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
ax = plt.subplot(2,3,5, projection='3d')
surf = ax.plot_surface(x,y,z,cmap = cm.coolwarm,edgecolor = 'none')
plt.colorbar(surf)
plt.tight_layout()
plt.show()

# bfs
def bfs(graph,start,goal,heur,path=[]):
  open = [(0,start)]
  close = set()
  close.add(start)
  while open:
    open.sort(key=lambda x:heur[x[1]],reverse=True)
    cost,node = open.pop()
    path.append(node)
    if node==goal:
      return cost,path
    close.add(node)
    for neigh,neigh_cost in graph[node]:
      if neigh not in close:
        close.add(node)
        open.append((cost+neigh_cost,neigh))
  return None
graph = {
    'A' : [('B',13),('C',5),('D',24)],
    'B' : [('A',13),('E',15)],
    'C' : [('A',5),('D',12),('E',10),('F',16)],
    'D' : [('A',24),('C',12),('F',4)],
    'E' : [('B',15),('C',10),('G',2)],
    'F' : [('C',16),('D',4),('G',1)],
    'G' : []
}
start = 'A'
goal = 'G'
heur = {
    'A':12,
    'B':10,
    'C':20,
    'D':5,
    'E':2,
    'F':21,
    'G':0
}
res = bfs(graph,start,goal,heur)
if res:
  print(f'the minimum cost path from {start} to {goal} is {res[1]}')
  print(f'the cost is{res[0]}')
else:
  print(f'no path from {start} to {goal}')
