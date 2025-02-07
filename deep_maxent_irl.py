import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#https://github.com/yrlu/irl-imitation/blob/master/deep_maxent_irl.py
#环境太简单了，不考虑

class DeepIRLFC(nn.Module):
    def __init__(self, n_input, lr, n_h1=400, n_h2=300, l2=10, device="cpu"):
        super(DeepIRLFC, self).__init__()
        self.device = device
        self.n_input = n_input
        self.lr = lr
        self.l2 = l2
        
        # Define the neural network
        self.fc1 = nn.Linear(n_input, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.reward_layer = nn.Linear(n_h2, 1)
        
        # Activation function
        self.elu = nn.ELU()

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2)

        # Move to device
        self.to(self.device)

    def forward(self, states):
        x = self.elu(self.fc1(states))
        x = self.elu(self.fc2(x))
        rewards = self.reward_layer(x)
        return rewards

    def get_rewards(self, states):
        """
        Get rewards for a batch of states.
        """
        states = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            rewards = self.forward(states)
        return rewards.cpu().numpy()

    def apply_grads(self, feat_map, grad_r):
        """
        Update the network's parameters using gradients.
        """
        self.optimizer.zero_grad()

        # Convert data to tensors
        feat_map = torch.FloatTensor(feat_map).to(self.device)
        grad_r = torch.FloatTensor(grad_r).view(-1, 1).to(self.device)

        # Compute the forward pass
        rewards = self.forward(feat_map)

        # Loss is the negative dot product of grad_r and rewards
        loss = -torch.sum(grad_r * rewards)

        # Backward pass
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)

        # Update parameters
        self.optimizer.step()

        return loss.item()

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
    """
    Compute the expected state visitation frequencies.
    返回p是1*N_STATES的数组，表示每个状态的访问频率
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = len(trajs[0])

    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    # Initialize visitation frequencies for t=0
    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] /= len(trajs)

    # Compute visitation frequencies for t > 0
    for t in range(T - 1):
        for s in range(N_STATES):
            if deterministic:
                mu[s, t + 1] = sum(
                    [mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)]
                )
            else:
                mu[s, t + 1] = sum(
                    [
                        sum(
                            [mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]
                        )
                        for pre_s in range(N_STATES)
                    ]
                )
    p = np.sum(mu, 1)
    return p


def demo_svf(trajs, n_states):
    """
    Compute state visitation frequencies from expert demonstrations.
    """
    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            p[step.cur_state] += 1
    p /= len(trajs)
    return p

def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters, device="cpu"):
    """
    Deep Maximum Entropy IRL algorithm implemented with PyTorch.
    P_a状态转移矩阵的作用：1.获取状态数量 2.作为env在value iteration中获取policy
    feature_map是N_STATES*input dim的矩阵，表示每个状态的特征(如何编码state有什么影响吗)        feat_map = np.eye(N_STATES)
    是否要手工设计state和action和状态转移矩阵（有env的话env提供状态转移矩阵）
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    # Initialize neural network，feat_map.shape[1]是input dim，n_h1=400, n_h2=300超参数
    nn_r = DeepIRLFC(feat_map.shape[1], lr, n_h1=400, n_h2=300, device=device)

    # Compute expert state visitation frequencies
    mu_D = demo_svf(trajs, N_STATES)

    # Training
    for iteration in range(n_iters):
        if iteration % (n_iters // 10) == 0:
            print(f"Iteration: {iteration}")

        # Compute the reward function
        rewards = nn_r.get_rewards(feat_map)

        # Compute the optimal policy using value iteration，是否deterministic policy影响下一步求我的奖励函数对应的likelihood
        _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

        # Compute expected state visitation frequencies
        mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)

        # Compute gradients on rewards
        grad_r = mu_D - mu_exp

        # Apply gradients to update the neural network
        loss = nn_r.apply_grads(feat_map, grad_r)

    # Final rewards
    rewards = nn_r.get_rewards(feat_map)
    return normalize(rewards)
