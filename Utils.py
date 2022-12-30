"""
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be

Source: Bashivan, et al."Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

Copyright (C) 2019 - UMons

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
"""

from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import numpy as np
from pylab import *
from scipy.interpolate import make_interp_spline
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def kfold(length, n_fold):
    tot_id = np.arange(length)  # arange函数返回一个有终点和起点的固定步长的排列
    np.random.shuffle(tot_id)  # 重新排序返回一个随机序列
    len_fold = int(length / n_fold)
    train_id = []
    test_id = []
    for i in range(n_fold):
        test_id.append(tot_id[i * len_fold:(i + 1) * len_fold])
        train_id.append(np.hstack([tot_id[0:i * len_fold], tot_id[(i + 1) * len_fold:-1]]))  # 返回从开始下标到结束下标之间的集合(不包括结束下标)
    return train_id, test_id


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""
    def __init__(self, label, image):
        self.label = label
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)

        return sample


def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1.plot(np.arange(niter), train_loss)
    # ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax1.plot(iters, costs,color='red',label=label_cost)
    ax2.plot(iters, accs,color='green',label=lable_acc)
    # plt.title(title, fontsize=24)
    # plt.xlabel("iter", fontsize=20)
    # plt.ylabel("acc(\%)", fontsize=20)
    # plt.plot(iters, costs,color='red',label=label_cost)
    # plt.plot(iters, accs,color='green',label=lable_acc)
    # 用于给图像加图例
    ax1.legend()
    ax2.legend()
    # 网格线设置
    plt.grid()
    plt.show()


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            # 上一个节点*0.8+当前节点*0.2
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            # 添加point
            smoothed_points.append(point)
    return smoothed_points

def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]



if __name__ == '__main__':
    data = pd.read_csv('D:\SeizuresCounter\loss_acc.csv')
    Macc = data[['Macc']]  # class 'pandas.core.frame.DataFrame'
    Mloss = data[['Mloss']]
    Macc = np.array(Macc)  # 将DataFrame类型转化为numpy数组
    Mloss = np.array(Mloss)

    Lacc = data[['Lacc']]
    Lloss = data[['Lloss']]
    Lacc = np.array(Lacc)  # 将DataFrame类型转化为numpy数组
    Lloss = np.array(Lloss)

    Tacc = data[['Tacc']]
    Tloss = data[['Tloss']]
    Tacc = np.array(Tacc)  # 将DataFrame类型转化为numpy数组
    Tloss = np.array(Tloss)

    macc = data[['macc']]
    mloss = data[['mloss']]
    macc = np.array(macc)  # 将DataFrame类型转化为numpy数组
    mloss = np.array(mloss)

    epochs = range(len(Macc))


    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('epochs', fontsize=18)
    ax1.set_ylabel('test accuracy(%)')
    ax2.set_ylabel('train loss')

    ax1.plot(smooth_xy(epochs, smooth_curve(Macc))[0], smooth_xy(epochs, smooth_curve(Macc))[1], color='red')
    ax1.plot(smooth_xy(epochs, smooth_curve(Lacc))[0], smooth_xy(epochs, smooth_curve(Lacc))[1], color='green')
    ax1.plot(smooth_xy(epochs, smooth_curve(Tacc))[0], smooth_xy(epochs, smooth_curve(Tacc))[1], color='olive')
    ax1.plot(smooth_xy(epochs, smooth_curve(macc))[0], smooth_xy(epochs, smooth_curve(macc))[1], color='blue')

    ax2.plot(smooth_xy(epochs, smooth_curve(Mloss))[0], smooth_xy(epochs, smooth_curve(Mloss))[1], color='red', linestyle='--')
    ax2.plot(smooth_xy(epochs, smooth_curve(Lloss))[0], smooth_xy(epochs, smooth_curve(Lloss))[1], color='green', linestyle='--')
    ax2.plot(smooth_xy(epochs, smooth_curve(Tloss))[0], smooth_xy(epochs, smooth_curve(Tloss))[1], color='olive', linestyle='--')
    ax2.plot(smooth_xy(epochs, smooth_curve(mloss))[0], smooth_xy(epochs, smooth_curve(mloss))[1], color='blue', linestyle='--')
    # ax1.legend()
    plt.title("acc&loss")
    plt.grid()
    plt.show()
