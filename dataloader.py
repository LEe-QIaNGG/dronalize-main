# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from argparse import Namespace

import torch
import numpy as np
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.collections import LineCollection

from lightning.pytorch import LightningDataModule
from torch_geometric.utils import subgraph
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as pyg_tf

from datamodules.dataset import DroneDataset
# from dataset import DroneDataset
from utils import import_from_module


class DroneDataModule(LightningDataModule):
    train: Dataset = None
    val: Dataset = None
    test: Dataset = None
    transform = None

    def __init__(self,
                 config: dict,
                 args: Namespace) -> None:
        super().__init__()
        self.root = config["root"]
        self.dataset = config["name"]
        self.batch_size = config["batch_size"]
        if config["transform"] is not None:
            if isinstance(config["transform"], list):
                self.transform = pyg_tf.Compose([import_from_module("datamodules.transforms",
                                                                    t)() for t in config["transform"]])
            else:
                self.transform = import_from_module("datamodules.transforms",
                                                    config["transform"])()

        self.small_data = args.small_ds
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train = DroneDataset(root=self.root, dataset=self.dataset, split='train',
                                  transform=self.transform, small_data=self.small_data)
        self.val = DroneDataset(root=self.root, dataset=self.dataset, split='val',
                                transform=self.transform, small_data=self.small_data)
        self.test = DroneDataset(root=self.root, dataset=self.dataset, split='val',
                                 transform=self.transform, small_data=self.small_data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    pl.seed_everything(42)


    def get_segments(pos, color):
        linefade = colors.to_rgb(color) + (0.0,)
        myfade = colors.LinearSegmentedColormap.from_list('my', [linefade, color])
        alphas = np.clip(np.exp(np.linspace(0, 1, pos.shape[0] - 1)) - 0.6, 0, 1)
        tmp = pos[:, :2][:, None, :]
        segments = np.hstack((tmp[:-1], tmp[1:]))
        return segments, alphas, myfade


    config = {'root': './data', 'name': 'highD', 'batch_size': 32,'transform':None}
    args = Namespace(small_ds=False, num_workers=0, pin_memory=False, persistent_workers=False)
    dm = DroneDataModule(config, args)
    dm.setup()

    gen = iter(dm.train_dataloader())
    data = next(gen)

    BATCH_IDX = 24  # 24 is used to create Fig. 1 in the paper

    batch = data['agent']['batch'] == BATCH_IDX
    pos = data['agent']['inp_pos'][batch]
    heading = data['agent']['inp_yaw'][batch]
    pos_eq_zero = pos == 0
    pos_eq_zero[0] = False
    pos[pos_eq_zero] = float("nan")

    gt = data['agent']['trg_pos'][batch]
    gt[gt == 0] = float("nan")

    valid_mask = data['agent']['valid_mask'][batch]
    ma_mask = data['agent']['ma_mask'][batch]
    ma_idx = torch.where(ma_mask[:, 0])[0]

    map_batch = data['map_point']['batch'] == BATCH_IDX
    map_pos = data['map_point']['position'][map_batch]
    map_type = data['map_point']['type'][map_batch]
    map_edge_index = data['map_point', 'to', 'map_point']['edge_index']
    map_edge_type = data['map_point', 'to', 'map_point']['type']

    map_edge_index, map_edge_type = subgraph(map_batch, map_edge_index,
                                             map_edge_type, relabel_nodes=True)

    #
    for i in range(map_edge_index.shape[1]):
        if map_edge_type[i] == 2:
            edge = map_edge_index[:, i]
            plt.plot(map_pos[edge, 0], map_pos[edge, 1], color='gray', lw=1,
                     zorder=1, alpha=.9, linestyle='solid')

        elif map_edge_type[i] == 1:
            edge = map_edge_index[:, i]
            plt.plot(map_pos[edge, 0], map_pos[edge, 1], color='darkgray', lw=0.5,
                     zorder=0, alpha=.6, linestyle=(0, (5, 10)))

    ax = plt.gca()

    COLOR = 'tab:red'
    for i in range(pos.shape[0]):
        if i == 0:
            COLOR = 'tab:blue'
        elif i in ma_idx:
            COLOR = 'tab:green'
        else:
            COLOR = 'tab:red'

        segments, alphas, myfade = get_segments(pos[i], COLOR)
        lc = LineCollection(segments, array=alphas, cmap=myfade, lw=5, zorder=0)
        line = ax.add_collection(lc)
        plt.plot(gt[i, :, 0], gt[i, :, 1], c=COLOR, marker='.', markersize=10, lw=2, alpha=0.3)

    ax.set_aspect('equal')
    ax.set_xlim(-50, 200)
    ax.set_ylim(-30, 35)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(data)
    
'''这个文件中的代码主要实现了一个名为 `DroneDataModule` 的数据模块，继承自 `LightningDataModule`，用于处理与无人机相关的数据集。以下是代码的主要功能和结构：

1. **导入必要的库**：代码导入了多个库，包括 PyTorch、NumPy、Matplotlib 和 PyTorch Geometric 等，用于数据处理和可视化。

2. **类定义**：定义了 `DroneDataModule` 类，负责加载和处理无人机数据集。

3. **初始化方法 (`__init__`)**：
   - 接收配置字典 `config` 和命名空间 `args` 作为参数。
   - 从配置中提取根目录、数据集名称、批量大小等信息。
   - 根据配置中的变换设置数据预处理。

4. **数据集设置 (`setup` 方法)**：
   - 根据不同的阶段（训练、验证、测试）加载数据集。
   - 使用 `DroneDataset` 类来创建训练、验证和测试数据集的实例。

5. **数据加载器方法**：
   - `train_dataloader`、`val_dataloader` 和 `test_dataloader` 方法分别返回训练、验证和测试数据的加载器，设置了批量大小、是否打乱数据等参数。

6. **主程序**：
   - 在 `__main__` 块中，设置随机种子以确保可重复性。
   - 定义了一个 `get_segments` 函数，用于生成线段和颜色渐变。
   - 创建了一个 `DroneDataModule` 实例，设置数据集，并获取训练数据。
   - 处理数据并使用 Matplotlib 绘制无人机轨迹和地图信息。

7. **可视化**：
   - 使用 Matplotlib 绘制无人机的轨迹和地图点，展示不同类型的边（如灰色和深灰色）以及无人机的位置和目标位置。

总结来说，这段代码的主要目的是定义一个数据模块，用于加载、处理和可视化无人机数据集，支持训练、验证和测试阶段的数据处理。
'''