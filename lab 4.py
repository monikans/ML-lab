# box plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = np.random.rand(100, 3)
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
plt.figure(figsize=(12, 8))
sns.heatmap(data = df)
plt.tight_layout()
plt.show()

#Alphabeta
def alpha_beta(depth,nodeindex,maximize,values,alpha,beta,path):
  if depth == 3:
    return values[nodeindex], path + [nodeindex]
  if maximize:
    best = float('-inf')
    best_path = []
    for i in range(2):
      val, newpath = alpha_beta(depth+1, nodeindex*2+i, False, values, alpha, beta, path + [nodeindex])
      if val > best:
        best = val
        best_path = newpath
        alpha = max(alpha,best)
        if alpha >= beta:
          break
    return best, best_path
  else:
    best = float('inf')
    best_path = []
    for i in range(2):
      val,newpath = alpha_beta(depth+1,nodeindex*2+i,True,values,alpha,beta,path+[nodeindex])
      if val < best:
        best = val
        best_path = newpath
        beta = min(best,beta)
        if alpha >= beta:
          break
    return best,best_path
values = [3,5,2,9,12,5,23,23]
optimal,opt_path = alpha_beta(0,0,True,values,float('-inf'),float('inf'),[])
print("optimal value is",optimal)
print('optimal path is',opt_path)
