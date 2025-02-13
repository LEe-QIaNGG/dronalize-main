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

import torch
import lightning.pytorch as pl

from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius, knn_graph
from torch_geometric.utils import to_undirected

from models.IRLmodel import IRLRewardModel,IRLTrajectoryPredictor
from SDD.sdd_functions import make_map_batch, make_map_batch_for_policy
from metrics import MinADE, MinFDE, MinAPDE, MissRate, CollisionRate


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, config: dict, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dataset = config["dataset"]
        self.max_epochs = config["epochs"]
        self.learning_rate = config["lr"]

        self.save_hyperparameters(ignore=['model'])

        self.optimizer_pose = torch.optim.AdamW(self.trajectory_predictor.parameters(), lr=self.learning_rate)
        self.optimizer_reward = torch.optim.AdamW(self.reward_model.parameters(), lr=self.learning_rate)

        self.min_ade = MinADE()
        self.min_fde = MinFDE()
        self.min_apde = MinAPDE()
        self.mr = MissRate()
        self.cr = CollisionRate()

    def post_process(self, data: HeteroData) -> HeteroData:
        pos = data['agent']['inp_pos'][:, -1]
        map_pos = data['map_point']['position']

        agent_batch = data['agent']['batch']
        map_batch = data['map_point']['batch']

        edge_index_a2a = knn_graph(x=pos, k=8, batch=agent_batch, loop=True)
        edge_index_a2a = to_undirected(edge_index_a2a)

        edge_index_m2a = radius(x=pos, y=map_pos, r=20, batch_x=agent_batch,
                                batch_y=map_batch, max_num_neighbors=8)

        data['agent']['edge_index'] = edge_index_a2a
        data['map', 'to', 'agent']['edge_index'] = edge_index_m2a
        return data

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.post_process(data)
        valid_mask = data['agent']['valid_mask']
        trg = data['agent']['trg_pos']

        pred = self.model(data)

        num_valid_steps = valid_mask.sum(-1)

        norm = torch.linalg.norm(pred - trg, dim=-1)

        masked_norm = norm * valid_mask

        scored_agents = num_valid_steps > 0

        summed_loss = masked_norm[scored_agents].sum(-1) / num_valid_steps[scored_agents]

        loss = summed_loss.mean()
        return loss, pred, trg

    def training_step(self, data: HeteroData) -> torch.Tensor:
        loss, _, trg = self(data)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 batch_size=trg.size(0), prog_bar=True)
        return loss

    def validation_step(self, data: HeteroData) -> None:
        ma_mask = data['agent']['ma_mask']
        ptr = data['agent']['ptr']

        loss, pred, trg = self(data)

        self.min_ade.update(pred, trg, mask=ma_mask)
        self.min_fde.update(pred, trg, mask=ma_mask)
        self.min_apde.update(pred, trg, mask=ma_mask)
        self.mr.update(pred, trg, mask=ma_mask)
        self.cr.update(pred, trg, ptr, mask=ma_mask)

        metric_dict = {"val_loss": loss,
                       "val_min_ade": self.min_ade,
                       "val_min_fde": self.min_fde,
                       "val_min_apde": self.min_apde,
                       "val_mr": self.mr,
                       "val_cr": self.cr}

        self.log_dict(metric_dict, on_step=False, on_epoch=True,
                      batch_size=trg.size(0), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class IRLLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module,config: dict):
        super().__init__()
        self.model = model
        self.dataset = config["dataset"]
        self.max_epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.grad_clip = config["grad_clip"]

        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        self.min_ade = MinADE()
        self.min_fde = MinFDE()
        self.min_apde = MinAPDE()
        self.mr = MissRate()
        self.cr = CollisionRate()

    def post_process(self, data: HeteroData) -> HeteroData:
        #就是为了获取agent to agent的边索引和map to agent的边索引
        pos = data['agent']['inp_pos'][:, -1]
        map_pos = data['map_point']['position']

        agent_batch = data['agent']['batch']
        map_batch = data['map_point']['batch']

        edge_index_a2a = knn_graph(x=pos, k=8, batch=agent_batch, loop=True)
        edge_index_a2a = to_undirected(edge_index_a2a)

        edge_index_m2a = radius(x=pos, y=map_pos, r=20, batch_x=agent_batch,
                                batch_y=map_batch, max_num_neighbors=8)

        data['agent']['edge_index'] = edge_index_a2a
        data['map', 'to', 'agent']['edge_index'] = edge_index_m2a
        return data

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.post_process(data)
        valid_mask = data['agent']['valid_mask']
        trg = data['agent']['trg_pos']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)
        map_to_agent_edge_index = data['map', 'to', 'agent']['edge_index']
        map_pos = data['map_point']['position']
        edge_index = data['agent']['edge_index']
        xoo_policy = self.model.sample(x, map_pos, self.model.init_state_enc)
        map_pos_policy =
        predictions_enc, predictions_dec, cost_pos_dec, cost_reward = self.model(x, map_pos, map_pos_policy, trg)

        # #xo, xp, xoo, xpo, did = batch 都是什么
        # xo = data['agent']['inp_pos']
        # xpo = data['agent']['trg_pos']
        # did = data['agent']['batch']
        # #mo 是当前批次（batch）的地图信息，表示与输入轨迹相关的地图数据，即s_t
        # mo = make_map_batch(xo, did, self.dataset.map, self.hparams.map_size).to(self.device)
        # #xoo_policy 是模型编码器（encoder）输出的预测轨迹。
        # xoo_policy = self.model.sample(xo.to(self.device), mo, self.model.init_state_enc)
        # #mo_policy 是基于模型预测轨迹 xoo_policy 生成的地图信息，即/hat{s}_{t}
        # mo_policy = make_map_batch_for_policy(xo, xo, xoo_policy, did, self.dataset.map, self.hparams.map_size).to(self.device)
        # predictions_enc, predictions_dec, cost_pos_dec, cost_reward = self.model(xoo.to(self.device), mo, mo_policy, xpo.to(self.device))

        return cost_pos_dec, cost_reward, predictions_dec,trg


    def training_step(self, data: HeteroData):
        optimizer_pose, optimizer_reward = self.optimizers()
        
        optimizer_pose.zero_grad()
        optimizer_reward.zero_grad()
        loss_pose, loss_reward,pred, trg = self(data)
        loss_pose.backward()
        optimizer_pose.step()
        loss_reward.backward()
        optimizer_reward.step()

        self.log("train_pose_loss", loss_pose, on_step=False, on_epoch=True,
             batch_size=trg.size(0), prog_bar=True)
        self.log("train_reward_loss", loss_reward, on_step=False, on_epoch=True,
             batch_size=trg.size(0), prog_bar=True)
        loss = loss_pose + loss_reward

        return loss


    def validation_step(self, data: HeteroData) -> None:
        ma_mask = data['agent']['ma_mask']
        ptr = data['agent']['ptr']

        loss_pose,lose_reward, pred, trg = self(data)

        self.min_ade.update(pred, trg, mask=ma_mask)
        self.min_fde.update(pred, trg, mask=ma_mask)
        self.min_apde.update(pred, trg, mask=ma_mask)
        self.mr.update(pred, trg, mask=ma_mask)
        self.cr.update(pred, trg, ptr, mask=ma_mask)

        metric_dict = {"val_loss_reward": loss_reward,
                        "val_pose_loss": loss_pose,
                       "val_min_ade": self.min_ade,
                       "val_min_fde": self.min_fde,
                       "val_min_apde": self.min_apde,
                       "val_mr": self.mr,
                       "val_cr": self.cr}

        self.log_dict(metric_dict, on_step=False, on_epoch=True,
                      batch_size=trg.size(0), prog_bar=True)

    def configure_optimizers(self):
        optimizer_pose = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer_reward = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return [
            {
                'optimizer': optimizer_pose,
                'gradient_clip_val': self.grad_clip,  # 第一个优化器的梯度裁剪阈值
                'gradient_clip_algorithm': 'norm'  # 使用 L2 范数裁剪
            },
            {
                'optimizer': optimizer_reward,
                'gradient_clip_val': self.grad_clip,  # 第二个优化器的梯度裁剪阈值
                'gradient_clip_algorithm': 'norm'  # 使用 L2 范数裁剪
            }
        ]

