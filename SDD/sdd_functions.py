'''
 FUNCTIONS SHARED BY EACH ALGORITHMS
 MADE BY DOOSEOP CHOI (d1024.choi@etri.re.kr)
 DATE : 2019-12-06
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import tensorflow as tf
import sys
#import matplotlib.pyplot as plt
import cv2
import copy

def _max(a,b):
    if (a>b):
        return a
    else:
        return b

def _min(a,b):
    if (a<b):
        return a
    else:
        return b

def map_roi_extract(x, y, map, width, fidx):
    """
    从地图中提取以 (x, y) 为中心的 ROI 区域。
    """
    size_row, size_col = map.shape[0], map.shape[1]
    x_center, y_center = int(x), int(y)

    # 如果 ROI 超出地图边界，返回全零的 ROI
    if (x_center - width < 0 or x_center + width - 1 > size_col - 1 or
        y_center - width < 0 or y_center + width - 1 > size_row - 1):
        part_map = torch.zeros((2 * width, 2 * width, 3), dtype=map.dtype, device=map.device)
    else:
        part_map = map[y_center - width:y_center + width, x_center - width:x_center + width, :].clone()

    # 如果 fidx == 0，返回全零的 ROI
    if fidx == 0:
        part_map = torch.zeros((2 * width, 2 * width, 3), dtype=map.dtype, device=map.device)

    return part_map


def make_map_batch(xo_batch, did_batch, maps, target_map_size):
    """
    为一批轨迹生成对应的 ROI 地图序列。
    """
    half_map_size = target_map_size // 2
    target_map = []

    for k in range(len(xo_batch)):
        xo = xo_batch[k]  # 当前轨迹
        map = maps[int(did_batch[k][0])]  # 当前地图
        map_seq = []

        for i in range(xo.shape[0]):
            x, y = xo[i, 0], xo[i, 1]
            corr_map = map_roi_extract(x, y, map, half_map_size, i)
            map_seq.append(corr_map)

        target_map.append(map_seq)

    return target_map


def make_map_batch_for_policy(xo_batch, xoo_batch, xoo_p_batch, did_batch, maps, target_map_size):
    """
    为策略生成的轨迹生成对应的 ROI 地图序列。
    """
    target_map = []

    for k in range(len(xo_batch)):
        xo = xo_batch[k]
        xoo = xoo_batch[k]
        xoo_p = xoo_p_batch[k]

        # 重构轨迹
        xoo_p_shift = torch.zeros_like(xoo_p)
        xoo_p_shift[1:] = xoo_p[:-1]
        xoo_p_shift[0, :] = xo[0, :]
        xoo_p_shift[1, :] = xoo[1, :]
        xoo_p_recon = torch.cumsum(xoo_p_shift, dim=0)

        # 当前地图
        map = maps[int(did_batch[k][0])]
        map_seq = []

        for i in range(xo.shape[0]):
            x, y = xoo_p_recon[i, 0], xoo_p_recon[i, 1]
            corr_map = map_roi_extract(x, y, map, target_map_size // 2, i)
            map_seq.append(corr_map)

        target_map.append(map_seq)

    return target_map


def weight_variable(shape, stddev=0.01):
    return nn.Parameter(torch.randn(shape) * stddev)


def bias_variable(shape, init=0.0):
    return nn.Parameter(torch.full(shape, init))


def conv_weight_variable(shape):
    if len(shape) < 4:
        stddev_xavier = math.sqrt(3.0 / (shape[0] + shape[1]))
    else:
        stddev_xavier = math.sqrt(3.0 / ((shape[0] * shape[1] * shape[2]) + (shape[0] * shape[1] * shape[3])))

    return nn.Parameter(torch.randn(shape) * stddev_xavier)


def conv_bias_variable(shape, init):
    return nn.Parameter(torch.full(shape, init))


def initialize_conv_filter(shape, name=None):
    W = conv_weight_variable(shape)
    b = conv_bias_variable(shape=[shape[0]], init=0.0)
    return W, b


def conv2d_strided_relu(x, W, b, strides, padding):
    conv = F.conv2d(x, W, bias=b, stride=strides, padding=padding)
    return F.relu(conv)


def max_pool(x, ksize, strides):
    return F.max_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


def shallow_convnet(input, w1, b1, w2, b2, w3, b3):
    conv1 = conv2d_strided_relu(input, w1, b1, strides=2, padding=0)
    conv2 = conv2d_strided_relu(conv1, w2, b2, strides=2, padding=0)
    conv3 = conv2d_strided_relu(conv2, w3, b3, strides=2, padding=0)
    output = torch.flatten(conv3, start_dim=1)
    return output


def calculate_reward(fwr, fbr, fc_in, cur_in):
    """
    计算奖励值。
    """
    state_vec = torch.cat([fc_in, cur_in], dim=1)
    reward = torch.sigmoid(F.linear(state_vec, fwr, fbr))
    return reward


def reshape_est_traj(x, batch_size, seq_length):
    """
    将预测的轨迹重新整形为 (batch_size, seq_length, 2)。
    """
    x_np = x.squeeze().cpu().numpy()
    x_reshape = [x_np[i * seq_length:(i + 1) * seq_length, :] for i in range(batch_size)]
    return x_reshape


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='>', epoch=10):
    """
    打印进度条。
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(length * iteration / float(total)))
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r [Epoch {epoch:02d}] {prefix} |{bar}| {percents}% {suffix}', end='')

    if iteration == total:
        print()