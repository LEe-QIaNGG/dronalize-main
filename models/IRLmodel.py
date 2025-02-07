
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as ptg
from torch_geometric.data import HeteroData

class IRLRewardModel(nn.Module):
    """奖励函数模块"""
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_hidden = config["num_hidden"]
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_index = data['agent']['edge_index']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)
        x = F.relu(self.fc1(x))
        reward = self.fc2(x)
        return reward

class IRLTrajectoryPredictor(nn.Module):
    """轨迹生成模块"""
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_hidden = config["num_hidden"]
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        next_state = self.fc2(x)
        return next_state