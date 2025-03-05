import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
  d = np.genfromtxt('log.txt', skip_header=13, skip_footer=4)
  print(f'efficiency: skr = {np.sum(d[:,1]>0)/d.shape[0]} fkr = {np.sum(d[:,2]>0)/d.shape[0]}')
  d = d[d[:,1]>0]
  d = d[d[:,2]>0]
  print(f'resolution: skr = {np.sqrt(np.average((d[:,1]-d[:,0])**2))} fkr = {np.sqrt(np.average((d[:,2]-d[:,0])**2))}')
