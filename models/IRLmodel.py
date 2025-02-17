
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
        self.ph = config["pred_hrz"]

        self.embed = nn.Linear(num_inputs, num_hidden)
        self.encoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.interaction = ptg.GATv2Conv(num_hidden, num_hidden, concat=False)
        self.decoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.output = nn.Linear(num_hidden, num_outputs)
        self.reward_function=nn.Linear(num_hidden,1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_index = data['agent']['edge_index']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)

        # map_to_agent_edge_index = data['map', 'to', 'agent']['edge_index']
        # map_pos = data['map_point']['position']

        x = self.embed(x)
        x_hat_t, h = self.encoder(x)
        x = h[-1]
        r_hat = self.reward_function(x_hat_t)
        r_groundTruth = self.reward_function(x)

        x = self.interaction(x, edge_index)
        x = x.unsqueeze(1).repeat(1, self.ph, 1)
        x, _ = self.decoder(x)

        pred = self.output(x)
        return pred, r_hat, r_groundTruth