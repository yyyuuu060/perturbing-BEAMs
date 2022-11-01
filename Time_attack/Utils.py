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
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc(\%)", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    # 用于给图像加图例
    plt.legend()
    # 网格线设置
    plt.grid()
    plt.show()


