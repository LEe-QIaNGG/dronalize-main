from sdd_functions import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sdd_functions import *


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Model(nn.Module):
    def __init__(self, config, infer=False):
        super(Model, self).__init__()
        
        # Parameter setting
        print('>> network configuration starts ...')
        
        # Define training or validation mode
        if infer:
            config.batch_size = 1
        
        self.args = config
        
        # In pose & out pose dim.
        self.input_dim = config["input_dim"]
        self.output_dim = config['input_dim']
        self.pred_length = config['pred_length']
        
        # Semantic map info
        self.map_size = config['map_size']
        
        # Convnet info
        self.conv_flat_size = 900
        
        # Reward function in & out info
        self.fc_size_in = self.conv_flat_size + self.input_dim
        self.fc_size_out = 1
        
        # Define network structure
        self.cell_enc = nn.LSTMCell(config['rnn_size'], config['rnn_size'])
        self.cell_dec = nn.LSTMCell(config['rnn_size'], config['rnn_size'])
        
        # Dropout layer
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        
        # Conv layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
        
        # Reward layer
        self.fc_reward = nn.Linear(self.fc_size_in, self.fc_size_out)
        
        # Embedding layers
        self.embedding_we = nn.Linear(self.input_dim, int(config['rnn_size'] / 2))
        self.embedding_wc = nn.Linear(self.conv_flat_size, int(config['rnn_size'] / 2))
        self.embedding_wcc = nn.Linear(config['rnn_size'], config['rnn_size'])
        self.embedding_wd = nn.Linear(self.input_dim, config['rnn_size'])
        
        # Output layers
        self.output_we = nn.Linear(config['rnn_size'], self.output_dim)
        self.output_wd = nn.Linear(config['rnn_size'], self.output_dim)
        
        # Initialize weights
        self.init_weights()
        
        print('>> network configuration is done ...')
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, gt_traj_enc, gt_map_enc, policy_map_enc, gt_traj_dec, output_keep_prob=1.0):
        # Process map info
        conv_out_gt_enc = self.process_map(gt_map_enc)
        conv_out_policy_enc = self.process_map(policy_map_enc)
        
        # Process pose info
        embedding_seqs_enc = self.process_pose(gt_traj_enc, conv_out_gt_enc)
        
        # Data goes through RNN Encoder
        predictions_enc, cost_reward, cost_policy = self.rnn_encoder(embedding_seqs_enc, gt_traj_enc, conv_out_gt_enc, conv_out_policy_enc)
        
        # Prediction by RNN Decoder
        predictions_dec, cost_pos_dec, cost_valid = self.rnn_decoder(gt_traj_enc, gt_traj_dec)
        
        # Define final cost and optimizer
        if not self.infer:
            cost_pos_dec /= (self.args.batch_size * self.args.pred_length)
            cost_valid /= (self.args.batch_size * self.args.pred_length)
            cost_reward /= self.args.batch_size
            cost_policy /= self.args.batch_size
            
            # L2 regularization
            l2_conv = self.args.lambda_param * sum(p.pow(2.0).sum() for p in self.conv_parameters() if p.requires_grad)
            l2_reward = self.args.lambda_param * sum(p.pow(2.0).sum() for p in self.reward_parameters() if p.requires_grad)
            l2_pose = self.args.lambda_param * sum(p.pow(2.0).sum() for p in self.pose_parameters() if p.requires_grad)
            
            cost_pos_dec += (self.args.gamma_param) * cost_policy + l2_pose + l2_conv
            cost_reward += l2_reward + l2_conv
        
        return predictions_enc, predictions_dec, cost_pos_dec, cost_reward
    
    def process_map(self, map_data):
        batch_size, seq_length, map_size, _, _ = map_data.size()
        map_data = map_data.view(batch_size * seq_length, map_size, map_size, 3)
        map_data = map_data.permute(0, 3, 1, 2)
        
        conv1 = torch.relu(self.conv1(map_data))
        conv2 = torch.relu(self.conv2(conv1))
        conv3 = torch.relu(self.conv3(conv2))
        
        conv_out = conv3.view(batch_size, seq_length, -1)
        return conv_out
    
    def process_pose(self, pose_data, conv_out):
        batch_size, seq_length, _ = pose_data.size()
        pose_data = pose_data.view(batch_size * seq_length, -1)
        
        embedding_pose = torch.relu(self.embedding_we(pose_data))
        embedding_conv = torch.relu(self.embedding_wc(conv_out.view(batch_size * seq_length, -1)))
        
        embedding_concat = torch.cat([embedding_pose, embedding_conv], dim=1)
        embedding_seqs = torch.relu(self.embedding_wcc(embedding_concat))
        
        embedding_seqs = embedding_seqs.view(batch_size, seq_length, -1)
        return embedding_seqs
    
    def rnn_encoder(self, embedding_seqs, gt_traj_enc, conv_out_gt_enc, conv_out_policy_enc):
        batch_size, seq_length, _ = embedding_seqs.size()
        predictions_enc = []
        cost_reward = 0.0
        cost_policy = 0.0
        
        for b in range(batch_size):
            cur_embed_seq = embedding_seqs[b]
            cur_gt_pose_seq = gt_traj_enc[b]
            cur_gt_convout_seq = conv_out_gt_enc[b]
            cur_policy_convout_seq = conv_out_policy_enc[b]
            
            reward_gt_avg = 0.0
            reward_policy_avg = 0.0
            prev_pred_pose = torch.zeros(1, self.input_dim)
            
            for f in range(seq_length):
                cur_embed_frm = cur_embed_seq[f].unsqueeze(0)
                cur_gt_convout_frm = cur_gt_convout_seq[f].unsqueeze(0)
                cur_policy_convout_frm = cur_policy_convout_seq[f].unsqueeze(0)
                cur_gt_pose = cur_gt_pose_seq[f].unsqueeze(0)
                
                if f == 0:
                    hx, cx = self.cell_enc(cur_embed_frm)
                else:
                    hx, cx = self.cell_enc(cur_embed_frm, (hx, cx))
                
                cur_pred_pose = self.output_we(hx)
                predictions_enc.append(cur_pred_pose)
                
                if f > 1:
                    reward_gt_avg += self.calculate_reward(cur_gt_convout_frm, cur_gt_pose)
                    reward_policy_avg += self.calculate_reward(cur_policy_convout_frm, prev_pred_pose)
                
                prev_pred_pose = cur_pred_pose
            
            reward_gt_avg /= (seq_length - 2)
            reward_policy_avg /= (seq_length - 2)
            cost_reward += -1.0 * torch.log(reward_gt_avg - reward_policy_avg + 1.0 + 1e-20)
            cost_policy += torch.log(reward_gt_avg - reward_policy_avg + 1.0 + 1e-20)
        
        return predictions_enc, cost_reward, cost_policy
    
    def rnn_decoder(self, gt_traj_enc, gt_traj_dec):
        batch_size, seq_length, _ = gt_traj_dec.size()
        predictions_dec = []
        cost_pos_dec = 0.0
        cost_valid = 0.0
        
        for b in range(batch_size):
            init_pose_dec = gt_traj_enc[b][-1].unsqueeze(0)
            cur_gt_traj_dec = gt_traj_dec[b]
            
            for f in range(seq_length):
                if f == 0:
                    cur_embed_frm = torch.relu(self.embedding_wd(init_pose_dec))
                    hx, cx = self.cell_dec(cur_embed_frm)
                else:
                    hx, cx = self.cell_dec(cur_embed_frm, (hx, cx))
                
                cur_pred_pose = self.output_wd(hx)
                predictions_dec.append(cur_pred_pose)
                
                cur_gt_pose = cur_gt_traj_dec[f].unsqueeze(0)
                mse_loss = torch.sum(torch.pow(cur_pred_pose - cur_gt_pose, 2.0))
                cost_pos_dec += mse_loss
                cost_valid += mse_loss
                
                cur_embed_frm = torch.relu(self.embedding_wd(cur_pred_pose))
        
        return predictions_dec, cost_pos_dec, cost_valid
    
    def calculate_reward(self, conv_out, pose):
        fc_input = torch.cat([conv_out, pose], dim=1)
        reward = self.fc_reward(fc_input)
        return reward
    
    def conv_parameters(self):
        return [p for n, p in self.named_parameters() if 'conv' in n]
    
    def reward_parameters(self):
        return [p for n, p in self.named_parameters() if 'fc_reward' in n]
    
    def pose_parameters(self):
        return [p for n, p in self.named_parameters() if 'embedding' in n or 'output' in n]
    
    def sample(self, xoo, mo, init_state_enc):
        self.eval()
        with torch.no_grad():
            feed = {
                'gt_traj_enc': xoo,
                'gt_map_enc': mo,
                'init_state_enc': init_state_enc,
                'output_keep_prob': 1.0
            }
            pred_offset = self.forward(**feed)
            est_offset = np.array(pred_offset).reshape(self.pred_length, self.args.input_dim)
        return est_offset
