# heat map
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = np.random.rand(100, 3)
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr,annot = True,cmap = 'coolwarm',vmin = -1,vmax = 1)
plt.tight_layout()
plt.show()

# minmax
def minmax(depth,nodeindex,maximize,values,alpha,beta,path):
  if depth == 3:
    return values[nodeindex], path + [nodeindex]
  if maximize:
    maxi = float('-inf')
    max_path = []
    for i in range(2):
      val, newpath = minmax(depth+1, nodeindex*2+i, False, values, alpha, beta, path + [nodeindex])
      if val > maxi:
        maxi = val
        max_path = newpath
    return maxi, max_path
  else:
    mini = float('inf')
    min_path = []
    for i in range(2):
      val,newpath = minmax(depth+1,nodeindex*2+i,True,values,alpha,beta,path+[nodeindex])
      if val < mini:
        mini = val
        mini_path = newpath
    return mini,mini_path
values = [3,5,2,9,12,5,23,23]
optimal,opt_path = minmax(0,0,True,values,float('-inf'),float('inf'),[])
print("optimal value is",optimal)
print('optimal path is',opt_path)
