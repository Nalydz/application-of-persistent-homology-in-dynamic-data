import numpy as np
import persim
import math
import math
import tadasets
import ripser
from ripser import ripser
from persim import plot_diagrams
from persim.persistent_entropy import *
from sklearn import datasets
import scipy
import scipy.stats as st
import pickle
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix


def floyd(W):
    n = np.size(W, 0)
    Distance_matrix=W.astype(int)
    RouteTable = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            RouteTable[i, j] = j
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if Distance_matrix[i, k]+Distance_matrix[k, j] < Distance_matrix[i, j]:
                    Distance_matrix[i, j] = Distance_matrix[i, k]+Distance_matrix[k, j]
                    RouteTable[i, j] = RouteTable[i, k]
    return Distance_matrix


a = np.loadtxt("H2009network.txt")
a = a.astype(int)
col = np.size(a, 0)
# 每1800秒钟生成一次
n_period = math.floor(212360 / 1800)
num = np.zeros(n_period, 1)
for i in range(n_period):
    N = np.full((114, 114), 116)
    for j in range(col):
        if i * 1800 < a[j, 0] <= (i + 1) * 1800:
            N[a[j, 1]-1, a[j, 2]-1] = 1
            N[a[j, 2]-1, a[j, 1]-1] = 1
    np.fill_diagonal(N, 0)
    N_1 = N_11 = floyd(N)
    N[N > 115] = 0
    N_1[N_1 > 115] = 0
    num[i] = 0
    if np.sum(N_1) > 0:
        num[i] = 1
        N_2 = csr_matrix(N_1)
        dig = ripser(N_2, distance_matrix=True)['dgms']
        plot_diagrams(dig)
        plt.savefig(
            'E:/pythonProject/figure/network_matrix_picture/network' + str(
                i) + '.png')
        plt.close()
        np.savetxt(
            'E:/pythonProject/data/network_matrix/network' + str(i) + '.csv',
            N_11, fmt='%i', delimiter=',')
        np.savetxt(
            'E:/pythonProject/data/network_matrix/network_floyd' + str(i) + '.txt',
            N_1, fmt='%i', delimiter=',')
        np.savetxt(
            'E:/pythonProject/data/network_matrix/network_' + str(
                i) + '.txt',
            N, fmt='%i', delimiter=',')
np.savetxt(
    'E:/pythonProject/data/network_matrix/num.txt',
    num, fmt='%i', delimiter=',')