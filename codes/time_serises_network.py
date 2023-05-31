import numpy as np
import persim
import math
import tadasets
import ripser
from ripser import ripser
from persim import plot_diagrams
from persim.persistent_entropy import *
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy
import scipy.stats as st
import pickle
from persim import PersistenceImager
from sklearn import preprocessing

# #计算时间序列网

# 读取时间序列信息并生成相应大小的矩阵
a = np.loadtxt("E:/pythonProject/data/temperature.txt")
row = 100
col = 1364
# 100个城市1364个月的温度数据
time_series = a.reshape(row, col)
# 每个period的跨度（月）
time_period = 60
# 两个跨度间的差别（月）
difference = 12
# 要生成的diagram数量
n_period = math.floor((col - 60) / difference)
# 存储计算好的diagram原始数据
s_diagram = []

#min_max_scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=1)

# 计算各个diagram
for i in range(n_period):
    period_time_series = time_series[:, i:i + time_period]
    # 距离矩阵
    N = np.sqrt(2 * (1 - np.corrcoef(period_time_series)))
    np.savetxt(
        "E:/pythonProject/data_statistics/time_network/coff/coff"+str(i)+".txt",
        N, fmt='%f', delimiter=',')
    # 计算距离矩阵行/列数
    N_row = np.size(N, 0)
    X_col = np.size(N, 1)
    # 滤流最大值
    filtration_value = np.max(N)
    # 计算persistent_diagram
    dig = ripser(N, thresh=filtration_value,
                 distance_matrix=True)['dgms']
    # 将输出的列表保存
    s_diagram.append(dig)
    plot_diagrams(dig)
    plt.savefig(
        'E:/pythonProject/figure/fig_project_time_network/diagram_' + str(
            i) + '.png')
    plt.close()
    H0_dgm = dig[0][dig[0][:, 1] != np.inf]
    H1_dgm = dig[1][dig[1][:, 1] != np.inf]
    dig = [H0_dgm, H1_dgm]
    pimgr = PersistenceImager(pixel_size=0.1)
    pimgr.fit(dig, skew=True)
    pimgs_1 = pimgr.transform(dig[1], skew=True)
    np.savetxt(
        'E:/pythonProject/data_statistics/time_network/image_matrix/pimgs_1_' + str(
            i) + '.txt',
        pimgs_1, fmt='%f', delimiter=',')
    pimgr.plot_image(pimgs_1)
    plt.savefig(
        'E:/pythonProject/figure/fig_project_time_network/image/image_1_' + str(
            i) + '.png')
    plt.close()




# 保存diagram原始数据
with open(
        'E:/pythonProject/data_statistics/time_network/s_diagram.pkl',
        'wb') as f:
    pickle.dump(s_diagram, f)

# 提取finite项并保存
s_diagram_finite = s_diagram
for i in range(len(s_diagram)):
    s_diagram_finite[i][0] = s_diagram[i][0][
        s_diagram[i][0][:, 1] != np.inf]
    s_diagram_finite[i][1] = s_diagram[i][1][
        s_diagram[i][1][:, 1] != np.inf]

# 保存处理好的diagram_finite形数据
with open(
        'E:/pythonProject/data_statistics/time_network/s_diagram_finite.pkl',
        'wb') as f:
    pickle.dump(s_diagram_finite, f)

# 数据规模
len_s_diagram = len(s_diagram)
# 生成两个distance的初始矩阵
was_distance_0 = np.zeros([len_s_diagram, len_s_diagram])
was_distance_1 = np.zeros([len_s_diagram, len_s_diagram])
bot_distance_0 = np.zeros([len_s_diagram, len_s_diagram])
bot_distance_1 = np.zeros([len_s_diagram, len_s_diagram])

for i in range(len(s_diagram)):
    for j in range(i + 1, len(s_diagram)):
        # 分维度计算wasserstein distance，注意提前将inf设置为有限值
        # 0维
        wasserstein_0, was_matching_0 = persim.wasserstein(
            s_diagram[i][0][
                s_diagram[i][0][:, 1] != np.inf],
            s_diagram[j][0][
                s_diagram[j][0][:, 1] != np.inf],
            matching=True)
        # 1维
        wasserstein_1, was_matching_1 = persim.wasserstein(
            s_diagram[i][1][
                s_diagram[i][1][:, 1] != np.inf],
            s_diagram[j][1][
                s_diagram[j][1][:, 1] != np.inf],
            matching=True)
        # 分维度计算bottleneck distance
        # 0维
        bottleneck_0, bot_matching_0 = persim.bottleneck(
            s_diagram[i][0][
                s_diagram[i][0][:, 1] != np.inf],
            s_diagram[j][0][
                s_diagram[j][0][:, 1] != np.inf],
            matching=True)
        # 1维
        bottleneck_1, bot_matching_1 = persim.bottleneck(
            s_diagram[i][1][
                s_diagram[i][1][:, 1] != np.inf],
            s_diagram[j][1][
                s_diagram[j][1][:, 1] != np.inf],
            matching=True)

        # 将数据写入矩阵
        was_distance_0[i][j] = wasserstein_0
        was_distance_0[j][i] = wasserstein_0
        was_distance_1[i][j] = wasserstein_1
        was_distance_1[j][i] = wasserstein_1

        bot_distance_0[i][j] = bottleneck_0
        bot_distance_0[j][i] = bottleneck_0
        bot_distance_1[i][j] = bottleneck_1
        bot_distance_1[j][i] = bottleneck_1

np.savetxt(
    "E:/pythonProject/data_statistics/time_network/was_distance_0.txt",
    was_distance_0, fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/was_distance_1.txt",
    was_distance_1, fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/bot_distance_0.txt",
    bot_distance_0, fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/bot_distance_0.txt",
    bot_distance_1, fmt='%f', delimiter=',')

# persistent entropy矩阵初始化
per_entropy = np.zeros([len_s_diagram, 2])
for i in range(len_s_diagram):
    # 计算各个维度的entropy并写入矩阵
    per_entropy[i][0] = persistent_entropy(s_diagram[i][0])
    per_entropy[i][1] = persistent_entropy(s_diagram[i][1])
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/per_entropy.txt",
    per_entropy, fmt='%f', delimiter=',')

# 其他统计计算选项
life_mean_diagram = np.zeros([len(s_diagram), 2])
life_max_diagram = np.zeros([len(s_diagram), 2])
life_standard_diagram = np.zeros([len(s_diagram), 2])
life_skewness_diagram = np.zeros([len(s_diagram), 2])
life_kurtosis_diagram = np.zeros([len(s_diagram), 2])
life_outstanding_diagram = np.zeros([len(s_diagram), 2])

location_mean_diagram = np.zeros([len(s_diagram), 2])
location_max_diagram = np.zeros([len(s_diagram), 2])
location_standard_diagram = np.zeros([len(s_diagram), 2])
location_skewness_diagram = np.zeros([len(s_diagram), 2])
location_kurtosis_diagram = np.zeros([len(s_diagram), 2])
location_entropy_diagram = np.zeros([len(s_diagram), 2])

for i in range(len(s_diagram)):
    # 去除inf项
    diagram_finite_0 = s_diagram[i][0][
        s_diagram[i][0][:, 1] != np.inf]
    diagram_finite_1 = s_diagram[i][1][
        s_diagram[i][1][:, 1] != np.inf]

    # ############life
    life_0 = diagram_finite_0[:, 1] - diagram_finite_0[:, 0]
    life_1 = diagram_finite_1[:, 1] - diagram_finite_1[:, 0]
    # life_mean
    life_mean_diagram[i][0] = np.mean(life_0)
    life_mean_diagram[i][1] = np.mean(life_1)
    # life_max
    life_max_diagram[i][0] = np.max(life_0)
    life_max_diagram[i][1] = np.max(life_1)
    # life_standard
    life_standard_diagram[i][0] = np.std(life_0)
    life_standard_diagram[i][1] = np.std(life_1)
    # life_skewness
    life_skewness_diagram[i][0] = st.skew(life_0)
    life_skewness_diagram[i][1] = st.skew(life_1)
    # life_kurtosis
    life_kurtosis_diagram[i][0] = st.kurtosis(life_0)
    life_kurtosis_diagram[i][0] = st.kurtosis(life_1)
    # life_outstanding
    ratio = 0.5
    life_outstanding_diagram[i][0] = np.sum(
        life_0 >= np.max(life_0) * ratio)
    life_outstanding_diagram[i][1] = np.sum(
        life_1 >= np.max(life_1) * ratio)

    # ############location
    location_0 = (diagram_finite_0[:, 1] + diagram_finite_0[
                                           :, 0]) / 2
    location_1 = (diagram_finite_1[:, 1] + diagram_finite_1[
                                           :, 0]) / 2
    # location_mean
    location_mean_diagram[i][0] = np.mean(location_0)
    location_mean_diagram[i][1] = np.mean(location_1)
    # location_max
    location_max_diagram[i][0] = np.max(location_0)
    location_max_diagram[i][1] = np.max(location_1)
    # location_standard
    location_standard_diagram[i][0] = np.std(location_0)
    location_standard_diagram[i][1] = np.std(location_1)
    # location_skewness
    location_skewness_diagram[i][0] = st.skew(location_0)
    location_skewness_diagram[i][1] = st.skew(location_1)
    # location_kurtosis
    location_kurtosis_diagram[i][0] = st.kurtosis(
        location_0)
    location_kurtosis_diagram[i][0] = st.kurtosis(
        location_1)

    # location_entropy(未考虑归一化)
    l_0 = np.sum(location_0)
    l_1 = np.sum(location_1)
    p_0 = 1 / l_0
    p_1 = 1 / l_1
    # entropy formula
    location_entropy_diagram[i][0] = -np.sum(
        p_0 * np.log(p_0))
    location_entropy_diagram[i][1] = -np.sum(
        p_1 * np.log(p_1))

# 保存各统计量
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_mean_diagram.txt",
    life_mean_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_max_diagram.txt",
    life_max_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_standard_diagram.txt",
    life_standard_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_skewness_diagram.txt",
    life_skewness_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_kurtosis_diagram.txt",
    life_kurtosis_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/life_outstanding_diagram.txt",
    life_outstanding_diagram,
    fmt='%f', delimiter=',')

np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_mean_diagram.txt",
    location_mean_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_max_diagram.txt",
    location_max_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_standard_diagram.txt",
    location_standard_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_skewness_diagram.txt",
    location_skewness_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_kurtosis_diagram.txt",
    location_kurtosis_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/time_network/location_entropy_diagram.txt",
    location_entropy_diagram,
    fmt='%f', delimiter=',')

'''
dets = np.loadtxt('E:/pythonProject/data_statistics/time_network/per_entropy.txt', delimiter=',')
p = np.arange(0, dets.shape[0])
plt.plot(1900+p, dets[:, 0])
plt.show()
'''
'''
print('0-bdis')
print(bot_distance_0)
print(('1-bdis'))
print(bot_distance_1)
print('0-was')
print(was_distance_0)
print('1-was')
print(was_distance_1)
print('entropy')
print(per_entropy)
'''
'''
data_clean = tadasets.dsphere(d=1, n=100, noise=0.0)
data_noisy = tadasets.dsphere(d=1, n=100, noise=0.1)
plt.scatter(data_clean[:,0], data_clean[:,1], label="clean data")
plt.scatter(data_noisy[:,0], data_noisy[:,1], label="noisy data")
plt.axis('equal')
plt.legend()
plt.show()
dgm_clean = ripser(data_clean)['dgms']
dgm_noisy = ripser(data_noisy)['dgms']
s_diagram.append(dgm_clean)
s_diagram.append(dgm_noisy)
persim.plot_diagrams([s_diagram[0][1], s_diagram[1][1]] , labels=['Clean $H_1$', 'Noisy $H_1$'])
plt.show()
distance_bottleneck, matching = persim.bottleneck(dgm_clean[1], dgm_noisy[1], matching=True)
persim.bottleneck_matching(s_diagram[0][1], s_diagram[1][1], matching, labels=['Clean $H_1$', 'Noisy $H_1$'])
plt.show()
print(distance_bottleneck)
'''
