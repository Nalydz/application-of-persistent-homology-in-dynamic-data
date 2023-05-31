import numpy as np
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from persim import plot_diagrams
import matplotlib.pyplot as plt
import scipy.stats as st
import pickle
from persim.persistent_entropy import *
from persim import PersistenceImager


def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray,
                 verbose: bool = True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )

    return y_embedded


max_embedding_dimension = 60
max_time_delay = 3
stride = 5
s_diagram = []
y = np.loadtxt("E:/pythonProject/data/Mallat_test.txt")
# 写入数据集
for i in range(np.size(y, 0)):
    y_periodic = y[i, :]
    embedder_periodic = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=max_time_delay,
        dimension=max_embedding_dimension,
        stride=stride,
    )

    y_periodic_embedded = embedder_periodic.fit_transform(y_periodic)
    #
    y_periodic_embedded = y_periodic_embedded[None, :, :]
    homology_dimensions = [0, 1, 2]
    # 计算3个维度的持续同调
    periodic_persistence = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions)
    d = periodic_persistence.fit_transform(y_periodic_embedded)
    diagrams = d[0]
    diagram_0 = diagrams[diagrams[:, 2] == 0][:, :2]
    diagram_1 = diagrams[diagrams[:, 2] == 1][:, :2]
    diagram_2 = diagrams[diagrams[:, 2] == 2][:, :2]
    dig = [diagram_0, diagram_1, diagram_2]
    s_diagram.append(dig)
    plot_diagrams(dig)
    plt.savefig(
        'E:/pythonProject/figure/Mallat_test/diagram_' + str(
            i) + '.png')
    plt.close()
    # 保存持续图表
    H0_dgm = diagram_0[diagram_0[:, 1] != np.inf]
    H1_dgm = diagram_1[diagram_1[:, 1] != np.inf]
    H2_dgm = diagram_2[diagram_2[:, 1] != np.inf]
    pdgms = [H0_dgm, H1_dgm, H2_dgm]
    pimgr = PersistenceImager(pixel_size=0.1)
    pimgr.fit(pdgms, skew=True)
    pimgs_1 = pimgr.transform(pdgms[1], skew=True)
    np.savetxt(
        'E:/pythonProject/data_statistics/Mallat_test/image_matrix/pimgs_1_' + str(
            i) + '.txt',
        pimgs_1, fmt='%f', delimiter=',')
    # 保存持续数字图像
    pimgr.plot_image(pimgs_1)
    plt.savefig(
        'E:/pythonProject/figure/Mallat_test/image/image_1_' + str(
            i) + '.png')
    plt.close()
    # 保存持续图像

# 保存diagram原始数据
with open(
        'E:/pythonProject/data_statistics/Mallat_test/s_diagram.pkl',
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
        'E:/pythonProject/data_statistics/Mallat_test/s_diagram_finite.pkl',
        'wb') as f:
    pickle.dump(s_diagram_finite, f)

# 数据规模
len_s_diagram = len(s_diagram)
# 生成两个distance的初始矩阵
was_distance_0 = np.zeros([len_s_diagram, len_s_diagram])
was_distance_1 = np.zeros([len_s_diagram, len_s_diagram])
bot_distance_0 = np.zeros([len_s_diagram, len_s_diagram])
bot_distance_1 = np.zeros([len_s_diagram, len_s_diagram])




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
    "E:/pythonProject/data_statistics/Mallat_test/life_mean_diagram.txt",
    life_mean_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/life_max_diagram.txt",
    life_max_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/life_standard_diagram.txt",
    life_standard_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/life_skewness_diagram.txt",
    life_skewness_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/life_kurtosis_diagram.txt",
    life_kurtosis_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/life_outstanding_diagram.txt",
    life_outstanding_diagram,
    fmt='%f', delimiter=',')

np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_mean_diagram.txt",
    location_mean_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_max_diagram.txt",
    location_max_diagram, fmt='%f',
    delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_standard_diagram.txt",
    location_standard_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_skewness_diagram.txt",
    location_skewness_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_kurtosis_diagram.txt",
    location_kurtosis_diagram,
    fmt='%f', delimiter=',')
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/location_entropy_diagram.txt",
    location_entropy_diagram,
    fmt='%f', delimiter=',')

# persistent entropy矩阵初始化
per_entropy = np.zeros([len_s_diagram, 2])
for i in range(len_s_diagram):
    # 计算各个维度的entropy并写入矩阵
    per_entropy[i][0] = persistent_entropy(s_diagram[i][0])
    per_entropy[i][1] = persistent_entropy(s_diagram[i][1])
np.savetxt(
    "E:/pythonProject/data_statistics/Mallat_test/per_entropy.txt",
    per_entropy, fmt='%f', delimiter=',')



























