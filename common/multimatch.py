# -*- coding: utf-8 -*-
"""
@Author  : zsy
@Time    : 2025/5/19 下午8:15
@File    : multimatch.py
@Desc    : 
"""
import numpy as np
import math
import collections

def cart2pol(x, y):
    """
    将笛卡尔坐标 (x, y) 转换为极坐标 (rho, theta)。

    参数:
    :param x: float，点的横坐标
    :param y: float，点的纵坐标

    返回:
    :return rho: float，点到原点的距离（半径）
    :return theta: float，点与x轴正方向的夹角（弧度）
    """
    # 计算点到原点的距离，使用勾股定理
    rho = np.sqrt(x ** 2 + y ** 2)
    # 计算夹角，使用 arctan2 考虑象限，返回弧度
    theta = np.arctan2(y, x)
    return rho, theta

def calcangle(x1, x2):
    """
    计算两个向量之间的夹角（角度制），常用于计算两次注视点间视线偏移的角度。

    参数:
    :param x1: list 或 np.array，向量1的坐标
    :param x2: list 或 np.array，向量2的坐标

    返回:
    :return angle: float，两个向量夹角，单位为度
    """
    # 先计算两个向量的点积，再除以两个向量的模长乘积，得到余弦值
    cos_angle = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    # 用反余弦函数求夹角，得到弧度
    angle_rad = math.acos(cos_angle)
    # 将弧度转换为角度
    angle = math.degrees(angle_rad)
    return angle

def gen_scanpath_structure(data):
    """
    将注视点数据（fixation vector）转换为基于向量的扫描路径（scanpath）表示形式。

    输入是一个 n×3 的注视向量：每一行表示 [起始x坐标, 起始y坐标, 注视持续时间]
    输出是一个有序字典，包含注视位置和由注视点之间计算得到的扫视向量信息。

    返回的结构包括：
    0: fixation_x        注视点 x 坐标
    1: fixation_y        注视点 y 坐标
    2: fixation_dur      注视时长
    3: saccade_x         扫视起点 x 坐标（即注视点）
    4: saccade_y         扫视起点 y 坐标
    5: saccade_lenx      扫视在 x 方向的长度
    6: saccade_leny      扫视在 y 方向的长度
    7: saccade_theta     扫视方向角（极角，弧度）
    8: saccade_rho       扫视距离（极径）
    """

    # 初始化各个字段的空列表
    fixation_x = []
    fixation_y = []
    fixation_dur = []
    saccade_x = []
    saccade_y = []
    saccade_lenx = []
    saccade_leny = []
    saccade_theta = []
    saccade_rho = []

    # 获取数据中注视点的个数  返回数据的形状（shape），即各维度的长度。 对于2D数组（如矩阵），形状是 (行数, 列数)。 对于 1D 数组，形状是 (长度,)。
    length = np.shape(data)[0]

    # 提取所有注视点的坐标和持续时间
    for i in range(0, length):
        fixation_x.append(data[i][0])  # 第i个注视点的x坐标
        fixation_y.append(data[i][1])  # 第i个注视点的y坐标
        fixation_dur.append(data[i][2])  # 第i个注视点的持续时间

    # 每次扫视连接的是当前注视点和下一个注视点。
    for i in range(0, length - 1):
        saccade_x.append(data[i][0])  # 第i段扫视的起点x
        saccade_y.append(data[i][1])  # 第i段扫视的起点y

    # 计算扫视向量的x/y方向长度和极坐标表示（rho, theta）
    for i in range(1, length):
        # 计算第i段扫视在x/y方向的长度（终点-起点）
        dx = fixation_x[i] - saccade_x[i - 1]
        dy = fixation_y[i] - saccade_y[i - 1]
        saccade_lenx.append(dx)
        saccade_leny.append(dy)

        # 将x/y方向的长度转换为极坐标形式（rho: 长度, theta: 方向）
        rho, theta = cart2pol(dx, dy)
        saccade_rho.append(rho)
        saccade_theta.append(theta)

    # 将所有计算结果整理为一个有序字典（保持顺序便于后续处理）
    # 会记住键的插入顺序，遍历时按插入顺序返回键值对。即使键的值被更新，顺序也不会改变（除非删除并重新插入）。
    eyedata = collections.OrderedDict()
    eyedata['fixation_x'] = fixation_x
    eyedata['fixation_y'] = fixation_y
    eyedata['fixation_dur'] = fixation_dur
    eyedata['saccade_x'] = saccade_x
    eyedata['saccade_y'] = saccade_y
    eyedata['saccade_lenx'] = saccade_lenx
    eyedata['saccade_leny'] = saccade_leny
    eyedata['saccade_theta'] = saccade_theta
    eyedata['saccade_rho'] = saccade_rho
    return eyedata

def keepsaccade(i,
                j,
                sim_lenx,
                sim_leny,
                sim_x,
                sim_y,
                sim_theta,
                sim_len,
                sim_dur,
                data
                ):
    """
    扫描路径简化辅助函数。在简化扫描路径（scanpath）过程中，当某一段扫视（saccade）不满足简化条件时，把它“原样保留”进简化结果。
    为什么要“简化扫描路径”？
    扫描路径包含大量注视点和扫视向量。
    有些扫视之间的角度相近、时间短、距离近，可以合并为一段。
    但有些扫视不能合并（比如方向差异太大、注视时间太长）——这些就要原样保留，以保证重要行为不被丢失。
    这个函数就是干这个事的：当遇到不能合并的扫视时，把它原样拷贝进简化路径。
    假设你在看一段人的眼动轨迹：
    有些扫视之间几乎重合、差别很小 —— 你可以简化合并。
    有些扫视忽然方向大变（比如从左上跳到右下）——不能简化，必须保留。
    :param i: 当前数据索引（原始数据）
    :param j: 当前数据索引（简化后数据）
    :param sim_lenx: 简化后扫视向量在 x 轴的分量列表
    :param sim_leny: 简化后扫视向量在 y 轴的分量列表
    :param sim_x: 简化后扫视起点 x 坐标列表
    :param sim_y: 简化后扫视起点 y 坐标列表
    :param sim_theta: 简化后扫视方向角度列表（极坐标）
    :param sim_len: 简化后扫视向量的幅度（模长）列表
    :param sim_dur: 简化后注视持续时间列表
    :param data: 原始扫描路径数据（通常是一个 OrderedDict）
    """
    # 原样保留当前扫视向量的 x 分量
    sim_lenx.insert(j, data['saccade_lenx'][i])
    # 原样保留当前扫视向量的 y 分量
    sim_leny.insert(j, data['saccade_leny'][i])
    # 原样保留当前扫视起点的 x 坐标
    sim_x.insert(j, data['saccade_x'][i])
    # 原样保留当前扫视起点的 y 坐标
    sim_y.insert(j, data['saccade_y'][i])
    # 原样保留当前扫视的方向角度（theta）
    sim_theta.insert(j, data['saccade_theta'][i])
    # 原样保留当前扫视向量的模长（rho）
    sim_len.insert(j, data['saccade_rho'][i])
    # 原样保留当前注视点的持续时间
    sim_dur.insert(j, data['fixation_dur'][i])

    # 索引后移，准备处理下一个扫视
    i += 1
    j += 1

    # 返回更新后的所有列表及索引
    return sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j

def simdir(data,
           TDir,
           TDur
           ):
    """
    基于扫视（saccade）方向之间的角度关系对扫描路径进行简化。
    如果两个连续的扫视之间的夹角小于阈值 TDir，且它们中间的注视点持续时间小于 TDur，
    则将这两个扫视合并为一个向量（等价于简化轨迹）。
    🎯 "把走得方向差不多、停留时间又不长的两步路，合成一步大路"

    :param data: 扫描路径数据（来自 gen_scanpath_structure 的输出）
    :param TDir: float，扫视方向角度阈值（单位：度）
    :param TDur: float，注视点持续时间阈值（单位：秒）
    :return: eyedata，简化后的一次扫描路径结构（有序字典）
    """
    # 没有扫视段，跳过
    if len(data['saccade_x']) < 1:
        return data
    else:
        # 初始化循环索引 i 和 j
        i = 0
        j = 0

        # 初始化用于保存简化后数据的空列表
        sim_dur = []     # 简化后的注视持续时间
        sim_x = []       # 扫视起点 x 坐标
        sim_y = []       # 扫视起点 y 坐标
        sim_lenx = []    # 扫视向量在 x 方向上的长度
        sim_leny = []    # 扫视向量在 y 方向上的长度
        sim_theta = []   # 扫视方向角度
        sim_len = []     # 扫视向量的模长（幅度）

        # 主循环：逐步检查并尝试简化每一对连续扫视
        while i <= len(data['saccade_x']) - 1:

            if i < len(data['saccade_x']) - 1:
                # 提取当前和下一个扫视向量
                v1 = [data['saccade_lenx'][i], data['saccade_leny'][i]]
                v2 = [data['saccade_lenx'][i + 1], data['saccade_leny'][i + 1]]
                # 计算它们之间的夹角
                angle = calcangle(v1, v2)
            else:
                # 最后一个扫视后没有下一个，设置为 ∞，不会进入合并逻辑
                angle = float('inf')

            # 如果夹角小于设定的方向阈值，且不是最后一个扫视
            if (angle < TDir) & (i < len(data['saccade_x']) - 1):
                # 如果中间注视点的持续时间小于设定阈值
                if data['fixation_dur'][i + 1] < TDur:
                    # 将两个扫视向量合并为一个新的向量
                    v_x = data['saccade_lenx'][i] + data['saccade_lenx'][i + 1]
                    v_y = data['saccade_leny'][i] + data['saccade_leny'][i + 1]
                    rho, theta = cart2pol(v_x, v_y)  # 转为极坐标：方向和幅度

                    # 存储新向量及其起点
                    sim_lenx.insert(j, v_x)
                    sim_leny.insert(j, v_y)
                    sim_x.insert(j, data['saccade_x'][i])
                    sim_y.insert(j, data['saccade_y'][i])
                    sim_theta.insert(j, theta)
                    sim_len.insert(j, rho)
                    sim_dur.insert(j, data['fixation_dur'][i])  # 保留第一个注视点的时长

                    # 跳过下一个点（因为已被合并），更新索引
                    i += 2
                    j += 1
                else:
                    # 中间注视点持续时间太长，无法合并，保留原始向量
                    sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                        i, j,
                        sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur,
                        data
                    )

            # 如果是最后一个扫视且角度也小，但中间注视仍很短
            elif (angle < TDir) & (i == len(data['saccade_x']) - 1):
                if data['fixation_dur'][i + 1] < TDur:
                    # 合并之前两个扫视（回溯处理）
                    v_x = data['saccade_lenx'][i - 2] + data['saccade_lenx'][i - 1]
                    v_y = data['saccade_leny'][i - 2] + data['saccade_leny'][i - 1]
                    rho, theta = cart2pol(v_x, v_y)

                    # 覆盖上一个合并项（修正）
                    sim_lenx[j - 1] = v_x
                    sim_leny[j - 1] = v_y
                    sim_theta[j - 1] = theta
                    sim_len[j - 1] = rho
                    # 合并持续时间（末尾点加上之前注视的一半）
                    sim_dur.insert(j, data['fixation_dur'][-1] + (data['fixation_dur'][i] / 2))

                    j -= 1
                    i += 1
                else:
                    # 不能合并，保留原始数据
                    sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                        i, j,
                        sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur,
                        data
                    )
            else:
                # 否则角度过大，不满足合并条件，保留原始扫视
                sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                    i, j,
                    sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur,
                    data
                )

        # 最后一个注视点的持续时间加入列表（循环外添加）
        sim_dur.append(data['fixation_dur'][-1])

        # 将所有结果打包为有序字典
        eyedata = collections.OrderedDict()
        eyedata['fixation_dur'] = sim_dur
        eyedata['saccade_x'] = sim_x
        eyedata['saccade_y'] = sim_y
        eyedata['saccade_lenx'] = sim_lenx
        eyedata['saccade_leny'] = sim_leny
        eyedata['saccade_theta'] = sim_theta
        eyedata['saccade_rho'] = sim_len

        return eyedata  # 返回简化后的扫描路径

def simlen(data, TAmp, TDur):
    """
    基于扫视长度进行扫描路径简化。

    如果两个连续的扫视满足：
    - 它们的长度都小于 TAmp（振幅阈值，像素单位）
    - 与其关联的注视时间小于 TDur（时间阈值，秒单位）

    则将它们合并为一个扫视向量，并调整相关数据。

    :param data: 字典格式的眼动数据，由 gen_scanpath_structure 返回
    :param TAmp: float，扫视长度阈值（像素）
    :param TDur: float，注视时长阈值（秒）
    :return: eyedata：简化后的眼动数据（有序字典）
    """

    # 如果扫视数据为空，则直接返回原数据
    if len(data['saccade_x']) < 1:
        return data
    else:
        # 初始化原数据索引和简化数据索引
        i = 0
        j = 0

        # 初始化用于存储简化后结果的列表
        sim_dur = []       # 注视持续时间
        sim_x = []         # 起始点 x 坐标
        sim_y = []         # 起始点 y 坐标
        sim_lenx = []      # 扫视 x 分量
        sim_leny = []      # 扫视 y 分量
        sim_theta = []     # 扫视方向角
        sim_len = []       # 扫视模长（幅度）

        # 主循环：遍历所有扫视向量
        while i <= len(data['saccade_x']) - 1:

            # 处理最后一个扫视向量
            if i == len(data['saccade_x']) - 1:

                # 如果最后一个扫视长度小于阈值
                if data['saccade_rho'][i] < TAmp:

                    # 如果当前注视或上一个注视时长短于阈值
                    if (data['fixation_dur'][-1] < TDur) or (data['fixation_dur'][-2] < TDur):

                        # 将最后两个扫视向量合并（向量相加）
                        v_x = data['saccade_lenx'][-2] + data['saccade_lenx'][-1]
                        v_y = data['saccade_leny'][-2] + data['saccade_leny'][-1]

                        # 转换为极坐标（模长和角度）
                        rho, theta = cart2pol(v_x, v_y)

                        # 替换倒数第二个向量（即合并后的结果）
                        sim_lenx[j - 1] = v_x
                        sim_leny[j - 1] = v_y
                        sim_theta[j - 1] = theta
                        sim_len[j - 1] = rho
                        sim_dur.insert(j, data['fixation_dur'][i - 1])

                        # 索引回退（因为两个合成了一个）
                        j -= 1
                        i += 1
                    else:
                        # 注视时间太长，不能合并，原样保留该扫视
                        sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                            i, j, sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, data)

                else:
                    # 如果最后一个扫视本身就太长，不能合并，原样保留
                    sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                        i, j, sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, data)

            # 处理中间的扫视向量（不是最后一个）
            else:

                # 如果当前扫视短于阈值
                if (data['saccade_rho'][i] < TAmp) and (i < len(data['saccade_x']) - 1):

                    # 且当前或下一个注视时间短于阈值
                    if (data['fixation_dur'][i + 1] < TDur) or (data['fixation_dur'][i] < TDur):

                        # 合并当前扫视与下一个扫视向量
                        v_x = data['saccade_lenx'][i] + data['saccade_lenx'][i + 1]
                        v_y = data['saccade_leny'][i] + data['saccade_leny'][i + 1]
                        rho, theta = cart2pol(v_x, v_y)

                        # 保存合并后的扫视向量和相关信息
                        sim_lenx.insert(j, v_x)
                        sim_leny.insert(j, v_y)
                        sim_x.insert(j, data['saccade_x'][i])
                        sim_y.insert(j, data['saccade_y'][i])
                        sim_theta.insert(j, theta)
                        sim_len.insert(j, rho)
                        sim_dur.insert(j, data['fixation_dur'][i])

                        # 跳过下一个，因为已合并
                        i += 2
                        j += 1
                    else:
                        # 注视时间太长，不能合并，原样保留该扫视
                        sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                            i, j, sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, data)

                else:
                    # 当前扫视长度太长，不能合并，原样保留
                    sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, i, j = keepsaccade(
                        i, j, sim_lenx, sim_leny, sim_x, sim_y, sim_theta, sim_len, sim_dur, data)

        # 最后一个注视时间单独加入（因为扫视是 n-1 个，注视是 n 个）
        sim_dur.append(data['fixation_dur'][-1])

        # 构建返回结果（有序字典）
        eyedata = collections.OrderedDict()
        eyedata['fixation_dur'] = sim_dur
        eyedata['saccade_x'] = sim_x
        eyedata['saccade_y'] = sim_y
        eyedata['saccade_lenx'] = sim_lenx
        eyedata['saccade_leny'] = sim_leny
        eyedata['saccade_theta'] = sim_theta
        eyedata['saccade_rho'] = sim_len

        return eyedata

def simplify_scanpath(data,
                      TAmp,
                      TDir,
                      TDur
                      ):
    """简化扫描路径（scanpaths），直到无法进一步简化为止。

    循环调用两个简化函数 simdir 和 simlen，直到扫描路径结构稳定。

    :param data: list of lists，有序字典结构，来自 gen_scanpath_structure 的输出
    :param TAmp: float，扫视幅度（长度）阈值，单位为像素
    :param TDir: float，扫视方向角度阈值，单位为度
    :param TDur: float，注视时长阈值，单位为秒

    :return: data：list of lists，已简化的向量形式扫描路径
    """
    looptime = 0  # 初始化简化迭代次数计数器

    while True:  # 进入无限循环，不断尝试简化直到满足终止条件
        data = simdir(data, TDir, TDur)  # 第一步：尝试方向角度（方向）上的简化
        data = simlen(data, TAmp, TDur)  # 第二步：尝试扫视幅度（长度）上的简化
        looptime += 1  # 每进行一次双重简化就计数+1

        # 当循环次数达到当前注视点数量时，认为无法进一步合并，结束简化
        if looptime == len(data['fixation_dur']):
            return data  # 返回已简化的扫描路径结构

def cal_vectordifferences(data1, data2):
    """
    计算两个扫描路径中所有扫视向量的欧几里得长度差，返回一个矩阵 M。

    对于 data1 中的每一个扫视向量，计算它与 data2 中所有扫视向量的差异（基于向量的模长差），
    构造一个差异矩阵 M，矩阵大小为 len(data1['saccade_x']) × len(data2['saccade_x'])
    计算两个人每一笔手势的“差异程度”，最终生成一个差异矩阵 告诉你“第一人的第1笔和第二人的所有笔有多像”、“第一人的第2笔和第二人的所有笔有多像”……依此类推。
    这种计算方式不是为了找"完全相同的第几笔"，而是为了回答：
    第一个人做的每一个动作，在第二个人的所有动作中，最接近的是哪个？
    （就像在人群中找最像你的人，不是只对比第一个，而是要比对所有人）
    问题：两个人看同一张图时，虽然眼球移动顺序不同，但是否会有相似的扫视模式？
    通过差异矩阵可以找到：
    人A的某个快速扫视（比如看标题）是否对应人B的某个类似扫视
    即使顺序不同，也能发现隐藏的相似片段

    :param data1: 字典格式的第一个眼动路径数据（向量表示）
    :param data2: 字典格式的第二个眼动路径数据（向量表示）
    :return: M: 差异矩阵（二维 NumPy 数组），表示每一对扫视向量的模长差
    """

    # 将第一个路径的 x 向量部分转为 NumPy 数组
    x1 = np.asarray(data1['saccade_lenx'])
    # 将第二个路径的 x 向量部分转为 NumPy 数组
    x2 = np.asarray(data2['saccade_lenx'])

    # 将第一个路径的 y 向量部分转为 NumPy 数组
    y1 = np.asarray(data1['saccade_leny'])
    # 将第二个路径的 y 向量部分转为 NumPy 数组
    y2 = np.asarray(data2['saccade_leny'])

    # 初始化两个空列表：row 存一行结果，M 最终是二维矩阵
    M = []   # 最终的矩阵
    row = [] # 每次循环构造的一行

    # 遍历 data1 中每一个扫视向量
    for i in range(0, len(x1)):
        # 将 data1 的第 i 个 x 向量扩展成与 x2 相同长度，用于向量化计算
        x_diff = abs(x1[i] * np.ones(len(x2)) - x2)

        # 同样对 y 分量进行向量化差值计算
        y_diff = abs(y1[i] * np.ones(len(y2)) - y2)

        # 使用欧几里得距离公式 sqrt(dx² + dy²) 计算所有对之间的模长差
        row.append(np.asarray(np.sqrt(x_diff ** 2 + y_diff ** 2)))

        # 每次加入 row（i 行）后，将其堆叠成矩阵 M（不断更新）
        M = np.stack(row)

    # 返回最终得到的差异矩阵
    return M

def createdirectedgraph(szM,
                        M,
                        M_assignment):
    """
    构建一个加权有向图，用于表示扫视向量差异矩阵 M 中所有可能的路径选择及其代价。下面用直观的方式解释它的设计逻辑和应用场景：

    参数：
    - szM: M 的形状，如 (行数, 列数)
    - M: 扫视向量模长差异矩阵（二维数组）
    - M_assignment: 节点编号矩阵，对 M 中每个元素按行优先顺序编号

    返回：
    - weightedGraph: 字典形式的有向图，每个键是当前节点，值是它可以到达的邻居节点及边权重（距离）
    """

    # 初始化两个字典，用于记录邻居节点（adjacent）和边的权重（weight）
    adjacent = {}
    weight = {}

    # 遍历 M 中的每个元素（即图中的每个节点），按行优先顺序
    for i in range(0, szM[0]):  # 行
        for j in range(0, szM[1]):  # 列
            # 当前节点在一维图中的编号
            currentNode = i * szM[1] + j

            # ========== 特殊位置的节点处理 ==========
            # 如果是最后一行，但不是最后一列，只能向右移动
            if (i == szM[0] - 1) & (j < szM[1] - 1):
                adjacent[M_assignment[i, j]] = [currentNode + 1]  # 向右连接
                weight[M_assignment[i, j]] = [M[i, j + 1]]        # 距离为右侧元素的值

            # 如果是最后一列，但不是最后一行，只能向下移动
            elif (i < szM[0] - 1) & (j == szM[1] - 1):
                adjacent[M_assignment[i, j]] = [currentNode + szM[1]]  # 向下连接
                weight[M_assignment[i, j]] = [M[i + 1, j]]              # 距离为下方元素的值

            # 如果是最后一个节点（右下角），不能再移动，只连接自己，权重为 0
            elif (i == szM[0] - 1) & (j == szM[1] - 1):
                adjacent[M_assignment[i, j]] = [currentNode]
                weight[M_assignment[i, j]] = [0]

            # 其余普通位置的节点，可以向右、向下、右下（对角）三个方向移动
            else:
                adjacent[M_assignment[i, j]] = [currentNode + 1,            # 向右
                                                currentNode + szM[1],       # 向下
                                                currentNode + szM[1] + 1]   # 向右下（对角）
                weight[M_assignment[i, j]] = [M[i, j + 1],                  # 对应的权重
                                              M[i + 1, j],
                                              M[i + 1, j + 1]]

    # ========== 构建最终嵌套字典结构（邻接表形式的图） ==========

    # 所有节点编号（从 0 到 M 中元素总数 - 1）
    Startnodes = range(0, szM[0] * szM[1])

    # 初始化：用于存储每个起始节点的所有邻居及权重对
    weightedEdges = []

    # 将邻接节点和对应的权重打包成元组对，如 [(1, 2.0), (3, 4.0)]
    for i in range(0, len(adjacent)):
        weightedEdges.append(list(zip(list(adjacent.values())[i],
                                      list(weight.values())[i])))

    # 初始化最终图结构
    weightedGraph = {}

    # 将每个起点节点与其邻居节点及权重建立字典映射
    for i in range(0, len(weightedEdges)):
        weightedGraph[Startnodes[i]] = dict(weightedEdges[i])

    # 返回最终图结构
    return weightedGraph

def dijkstra(weightedGraph,
             start,
             end):
    """
    dijkstra() 函数与 createdirectedgraph() 的组合实现了 扫视轨迹的动态时间规整（DTW），其核心意义是：
    找到两条眼动轨迹的最优匹配方案将
    眼动轨迹的复杂比对问题转化为可计算的最短路径问题，最终输出人类可解释的对齐方案和相似性评分，为行为分析提供量化基础。
    输入：
    差异矩阵 M（cal_vectordifferences() 生成）
    矩阵中的每个值 M[i][j] 表示 data1 的第 i 个扫视与 data2 的第 j 个扫视的差异程度
    输出：
    最短路径：表示两条轨迹扫视片段的最优对应关系
    总代价：量化两条轨迹的整体相似性（值越小越相似）

    使用 Dijkstra 算法，在加权有向图中寻找从 start 节点到 end 节点的最短路径。

    解决扫视顺序不一致问题:不同用户看同一图片时，扫视顺序可能不同（例如：A先看标题，B先看图）通过动态对齐，找到实质相似的扫视片段，忽略顺序差异

    参数：
    - weightedGraph: dict，嵌套字典结构，键是节点，值是邻居节点及对应权重
      例如：{0: {1: 10, 2: 5}, 1: {3: 1}, ...}
    - start: int，起点节点编号，通常是 0
    - end: int，终点节点编号，通常是 M 矩阵的最后一个元素索引

    返回：
    - path: list，起点到终点的最短路径节点序列（包含起点和终点）
    - dist: float，最短路径的总代价（路径上所有边权重之和）
    """

    # 初始化距离字典 dist，存储从 start 到每个节点的当前已知最短距离
    dist = {}
    # 初始化前驱字典 pred，用于记录最短路径中每个节点的前驱节点
    pred = {}

    # 需要评估的节点集合，即所有节点的键
    to_assess = weightedGraph.keys()

    # 将所有节点的初始距离设为无穷大，表示尚未访问过
    # 前驱节点初始设为 None，表示尚未确定路径
    for node in weightedGraph:
        dist[node] = float('inf')
        pred[node] = None

    # sp_set 用于存储已经找到最短距离的节点（已确定最短路径的节点）
    sp_set = []

    # 起点距离设为 0，保证算法从这里开始
    dist[start] = 0

    # 当尚未确定所有节点最短距离时，继续循环
    while len(sp_set) < len(to_assess):
        # 从未确定最短距离的节点中筛选出当前距离最小的节点
        still_in = {node: dist[node] for node in [node for node in to_assess if node not in sp_set]}
        closest = min(still_in, key=dist.get)  # 找出距离最小的节点

        # 将该节点标记为已确定最短路径
        sp_set.append(closest)

        # 遍历该节点所有邻居节点，尝试松弛操作，更新最短距离和前驱节点
        for node in weightedGraph[closest]:
            # 若经过 closest 节点到达 node 的距离更短，则更新
            if dist[node] > dist[closest] + weightedGraph[closest][node]:
                dist[node] = dist[closest] + weightedGraph[closest][node]
                pred[node] = closest

    # 反向构建路径，从终点开始往回追溯前驱节点
    path = [end]
    while start not in path:
        path.append(pred[path[-1]])

    # 将路径反转，保证路径从 start 到 end 的顺序
    return path[::-1], dist[end]

def cal_angulardifference(data1, data2, path, M_assignment):
    """
    之前 dijkstra 的差异矩阵 M 只考虑扫视长度差异 此函数增加方向差异分析，形成更全面的比对维度
    计算两条扫描路径中，每对配对扫视向量的角度差异。
    在已通过 dijkstra() 找到最佳匹配路径的基础上，进一步分析：
    每个匹配对的方向差异：data1 的第 i 个扫视 vs data2 的第 j 个扫视的角度差
    输出：一组弧度值，表示每对匹配扫视的运动方向偏离程度（0表示完全同向，π表示完全反向）

    参数：
    - data1, data2：两个扫描路径的向量化表示，字典格式，包含每个扫视的角度'saccade_theta'
    - path：最佳匹配路径索引数组，表示哪对扫视向量是配对的
    - M_assignment：矩阵，表示配对对应关系的索引矩阵

    返回：
    - anglediff：数组，存放每对配对扫视的角度差异（弧度）
    """

    # 从两个扫描路径中取出每个扫视的角度数组
    theta1 = data1['saccade_theta']
    theta2 = data2['saccade_theta']

    anglediff = []
    # 遍历每个配对路径索引
    for k in range(len(path)):
        # 找出该配对索引在M_assignment中的对应位置
        i, j = np.where(M_assignment == path[k])
        # 取出对应两扫视的角度
        spT = [theta1[i.item()], theta2[j.item()]]
        # 将角度调整到 -pi 到 pi 范围内（负角度转换）
        for t in range(len(spT)):
            if spT[t] < 0:
                spT[t] = math.pi + (math.pi + spT[t])
        # 计算两角度的绝对差
        spT = abs(spT[0] - spT[1])
        # 角度差如果大于pi，则取补角（2pi - 差值）
        if spT > math.pi:
            spT = 2 * math.pi - spT
        # 添加到结果列表
        anglediff.append(spT)

    return anglediff

def cal_durationdifference(data1, data2, path, M_assignment):
    """
    计算两条扫描路径中，每对配对注视的持续时间差异（归一化差异）。
    每对匹配注视点的持续时间差异（data1的第i个注视 vs data2的第j个注视）
    反映两人在每个关键区域的注意力分配差异程度
    参数同上。

    返回：
    - durdiff：数组，存放每对配对注视持续时间的归一化绝对差（0~1之间）
    """

    dur1 = data1['fixation_dur']
    dur2 = data2['fixation_dur']

    durdiff = []
    for k in range(len(path)):
        i, j = np.where(M_assignment == path[k])
        maxlist = [dur1[i.item()], dur2[j.item()]]
        # 计算两个持续时间的绝对差，并用两者中较大的持续时间归一化
        durdiff.append(abs(dur1[i.item()] - dur2[j.item()]) / abs(max(maxlist)))

    return durdiff

def cal_lengthdifference(data1, data2, path, M_assignment):
    """
    计算两条扫描路径中，每对配对扫视的长度差异。
    这个函数是眼动分析中的基础物理指标提取工具，通过量化扫视长度差异，为界面设计优化和用户行为分析提供客观依据。
    返回：
    - lendiff：数组，存放每对配对扫视长度的绝对差
    """

    len1 = np.asarray(data1['saccade_rho'])
    len2 = np.asarray(data2['saccade_rho'])

    lendiff = []
    for k in range(len(path)):
        i, j = np.where(M_assignment == path[k])
        lendiff.append(abs(len1[i] - len2[j]))

    return lendiff

def cal_positiondifference(data1, data2, path, M_assignment):
    """
    计算两条扫描路径中，每对配对扫视终点的空间位置差异。

    返回：
    - posdiff：数组，存放每对配对扫视终点的欧氏距离
    """

    x1 = np.asarray(data1['saccade_x'])
    x2 = np.asarray(data2['saccade_x'])
    y1 = np.asarray(data1['saccade_y'])
    y2 = np.asarray(data2['saccade_y'])

    posdiff = []
    for k in range(len(path)):
        i, j = np.where(M_assignment == path[k])
        # 计算二维坐标的欧氏距离
        posdiff.append(math.sqrt((x1[i.item()] - x2[j.item()]) ** 2 +
                                 (y1[i.item()] - y2[j.item()]) ** 2))

    return posdiff

def cal_vectordifferencealongpath(data1, data2, path, M_assignment):
    """
    计算两条扫描路径中，每对配对扫视向量（x,y分量）的差异。

    返回：
    - vectordiff：数组，存放每对配对扫视向量的欧氏距离差
    """

    x1 = np.asarray(data1['saccade_lenx'])
    x2 = np.asarray(data2['saccade_lenx'])
    y1 = np.asarray(data1['saccade_leny'])
    y2 = np.asarray(data2['saccade_leny'])

    vectordiff = []
    for k in range(len(path)):
        i, j = np.where(M_assignment == path[k])
        # 计算xy分量差的欧氏距离
        vectordiff.append(np.sqrt((x1[i.item()] - x2[j.item()]) ** 2 +
                                  (y1[i.item()] - y2[j.item()]) ** 2))

    return vectordiff

def getunnormalised(data1,
                    data2,
                    path,
                    M_assignment):
    """
    计算五个维度上的未归一化（unnormalised）扫描路径相似度。

    你可以想象两个“眼动轨迹”（扫描路径），是两条“折线”——每个点代表注视位置，每段线段表示扫视方向。
    这段函数就是：
    沿着这两条折线 最优配对的线段对（由 path 和 M_assignment 给出），逐对比较它们的差异，在五个角度上看它们是否“长得像”。

    函数会调用五个分别计算向量差异、角度差异、长度差异、位置差异和持续时间差异的函数，
    并对每个维度对应路径上的所有差异值取中位数，作为该维度的未归一化相似度指标。

    参数：
    - data1: array-like，第一个扫描路径的向量化表示（列表或数组）
    - data2: array-like，第二个扫描路径的向量化表示
    - path: array-like，最优路径的节点索引数组，表示两个扫描路径中匹配的扫视对
    - M_assignment: array-like，矩阵 M 的索引矩阵，范围从 0 到 M 中元素总数，用于定位配对

    返回：
    - unnormalised: array，五个维度的未归一化相似度数组，顺序为：
      向量形状差异（VecSim），角度差异（DirSim），长度差异（LenSim），
      位置差异（PosSim），持续时间差异（DurSim）

    示例：
    >>> unorm_res = getunnormalised(scanpath_rep1, scanpath_rep2, path, M_assignment)
    """

    # 将输入参数打包，方便传给下面五个计算函数
    args = data1, data2, path, M_assignment

    # 计算路径上的向量形状差异，返回一个数组，取中位数作为整体相似度指标
    VecSim = np.median(cal_vectordifferencealongpath(*args))

    # 计算路径上的角度差异，取中位数作为相似度指标
    DirSim = np.median(cal_angulardifference(*args))

    # 计算路径上的长度差异，取中位数作为相似度指标
    LenSim = np.median(cal_lengthdifference(*args))

    # 计算路径上的位置差异，取中位数作为相似度指标
    PosSim = np.median(cal_positiondifference(*args))

    # 计算路径上的持续时间差异，取中位数作为相似度指标
    DurSim = np.median(cal_durationdifference(*args))

    # 将五个维度的未归一化相似度汇总成一个数组
    unnormalised = [VecSim, DirSim, LenSim, PosSim, DurSim]

    # 返回该数组
    return unnormalised

def normaliseresults(unnormalised,
                     sz=[1280, 720]
                     ):
    """
    归一化相似度指标，将未归一化的差异值转换为 0 到 1 之间的相似度分数，
    其中 1 表示完全相似，0 表示最不相似。

    归一化策略：
    - 向量相似度（VectorSimilarity）：除以屏幕对角线的两倍（最大可能距离）
    - 方向相似度（DirectionSimilarity）：除以 π （最大角度差）
    - 长度相似度（LengthSimilarity）：除以屏幕对角线长度
    - 位置相似度（PositionSimilarity）：除以屏幕对角线长度
    - 持续时间相似度（DurationSimilarity）：已预先归一化，无需再次处理

    参数：
    - unnormalised: array，五个维度的未归一化相似度，来自 getunnormalised() 函数的输出
    - sz: list，屏幕尺寸，默认宽1280像素，高720像素，用于计算屏幕对角线长度

    返回：
    - normalresults: array，归一化后的五个相似度指标数组

    示例：
    >>> normal_res = normaliseresults(unnormalised, sz = [1280, 720])
    """

    # 计算屏幕对角线长度
    screen_diag = math.sqrt(sz[0] ** 2 + sz[1] ** 2)

    # 向量相似度 = 1 - (未归一化向量差 / 2倍屏幕对角线)
    # 理由是向量差最大可达到屏幕对角线的两倍，越小差异越大，相似度越接近1
    VectorSimilarity = 1 - unnormalised[0] / (2 * screen_diag)

    # 方向相似度 = 1 - (未归一化方向差 / π)
    # 方向最大差为180度（π弧度），相似度取反
    DirectionSimilarity = 1 - unnormalised[1] / math.pi

    # 长度相似度 = 1 - (未归一化长度差 / 屏幕对角线)
    LengthSimilarity = 1 - unnormalised[2] / screen_diag

    # 位置相似度 = 1 - (未归一化位置差 / 屏幕对角线)
    PositionSimilarity = 1 - unnormalised[3] / screen_diag

    # 持续时间相似度 = 1 - 未归一化持续时间差（已预先归一化，无需额外处理）
    DurationSimilarity = 1 - unnormalised[4]

    # 汇总归一化结果
    normalresults = [VectorSimilarity, DirectionSimilarity, LengthSimilarity,
                     PositionSimilarity, DurationSimilarity]

    # 返回归一化后的五个相似度指标
    return normalresults

def docomparison(fixation_vectors1,
                 fixation_vectors2,
                 sz=[1280, 720],
                 grouping=False,
                 TDir=0.0,
                 TDur=0.0,
                 TAmp=0.0):
    """
    比较两个扫描路径（Scanpath）在五个维度上的相似性：形状（Shape）、方向（Angle）、
    长度（Length）、位置（Position）、持续时间（Duration）。

    参数：
    - fixation_vectors1, fixation_vectors2: 每个是 n×3 的注视向量序列（x, y, duration）
    - sz: 屏幕尺寸，单位为像素，默认是 [1280, 720]
    - grouping: 是否启用扫视合并简化（根据阈值）
    - TDir: 简化时使用的角度阈值（度）
    - TDur: 简化时使用的持续时间阈值（秒）
    - TAmp: 简化时使用的扫视长度阈值（像素）

    返回：
    - scanpathcomparisons: 包含五个相似性度量的列表，取值在 0~1 之间，越接近 1 越相似
    """

    # 初始化用于存储最终相似性结果的列表
    scanpathcomparisons = []

    # 如果两个注视路径都至少包含3个注视点（即至少2次扫视），才能进行比较
    if (len(fixation_vectors1) >= 3) & (len(fixation_vectors2) >= 3):

        # 将两个原始注视向量数据转换成几何结构（计算扫视向量、角度等）
        subj1 = gen_scanpath_structure(fixation_vectors1)
        subj2 = gen_scanpath_structure(fixation_vectors2)

        # 如果启用了 grouping（即简化开关为真），则按设定阈值进行扫视简化
        if grouping:
            subj1 = simplify_scanpath(subj1, TAmp, TDir, TDur)
            subj2 = simplify_scanpath(subj2, TAmp, TDir, TDur)

        # 计算 subj1 和 subj2 所有扫视向量之间的模长差（欧几里得距离），生成差异矩阵 M
        M = cal_vectordifferences(subj1, subj2)

        # 获取差异矩阵 M 的形状，用于生成结点编号
        szM = np.shape(M)

        # 差异矩阵 M 中的每个位置（即每个扫视对）分配一个唯一编号
        M_assignment = np.arange(szM[0] * szM[1]).reshape(szM[0], szM[1])

        # 使用差异矩阵 M 和编号矩阵 M_assignment 构造加权有向图
        weightedGraph = createdirectedgraph(szM, M, M_assignment)

        # 使用 Dijkstra 算法在加权图中寻找从起点到终点的最短路径（代价最小的匹配）
        path, dist = dijkstra(weightedGraph, 0, szM[0] * szM[1] - 1)

        # 使用该最优路径对两个路径进行对齐，并提取未归一化的相似度指标（五个维度）
        unnormalised = getunnormalised(subj1, subj2, path, M_assignment)

        # 将未归一化指标根据屏幕尺寸等进行归一化，得到最终的五维相似度分数（0~1）
        normal = normaliseresults(unnormalised, sz)

        # 将五维相似性结果加入最终结果列表
        scanpathcomparisons.append(normal)

    else:
        # 如果任一注视路径太短，无法比较，返回包含 5 个 NaN 的向量
        scanpathcomparisons.append(np.repeat(np.nan, 5))

    # 返回最终结果
    return scanpathcomparisons
