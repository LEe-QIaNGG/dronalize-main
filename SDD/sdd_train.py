import argparse
import os
import pickle
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于进度条显示

from sdd_model import Model  # 假设你已经将模型转换为 PyTorch 实现
from sdd_utils import SDDDataset  # 假设你已经将数据加载逻辑转换为 PyTorch 的 Dataset
from sdd_functions import make_map_batch, make_map_batch_for_policy  # 假设这些函数已经适配 PyTorch


def print_training_info(args):
    print('-----------------------------------------------------------')
    print('------------ Encoder-Decoder LSTM INFORMATION -------------')
    print(f'.dataset_num {args.dataset_num}, exp id {args.exp_id}')
    print('.network structure: LSTM')
    print(f'   rnn size ({args.rnn_size})')
    print(f'   num layers ({args.num_layers})')
    print('.training setting')
    print(f'   num epochs ({args.num_epochs})')
    print(f'   batch size ({args.batch_size})')
    print(f'   obs_length ({args.obs_length})')
    print(f'   pred_length ({args.pred_length})')
    print(f'   map size ({args.map_size})')
    print(f'   learning rate ({args.learning_rate:.5f})')
    print(f'   reg. lambda ({args.lambda_param:.5f})')
    print(f'   gamma param ({args.gamma_param:.5f})')
    print(f'   grad_clip ({args.grad_clip:.2f})')
    print(f'   keep prob ({args.keep_prob:.2f})')
    print(f'   load pretrained ({args.load_pretrained})')
    print(f'   start epoch ({args.start_epoch})')
    print(f'   data load step ({args.data_load_step})')
    print('------------------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()

    # Network structure
    parser.add_argument('--rnn_size', type=int, default=256, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--input_dim', type=int, default=2, help='dimension of input vector')

    # Training setting
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='dataset path')
    parser.add_argument('--exp_id', type=int, default=0, help='experiment id')
    parser.add_argument('--dataset_num', type=int, default=0, help='dataset number')
    parser.add_argument('--gpu_num', type=int, default=0, help='GPU device id')
    parser.add_argument('--load_pretrained', type=int, default=0, help='load pre-trained network')
    parser.add_argument('--batch_size', type=int, default=4, help='minibatch size')
    parser.add_argument('--obs_length', type=int, default=8, help='observation sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save', help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=1.5, help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lambda_param', type=float, default=0.0001, help='L2 regularization weight')
    parser.add_argument('--gamma_param', type=float, default=0.001, help='IRL regularization weight')
    parser.add_argument('--keep_prob', type=float, default=0.8, help='dropout keep probability')
    parser.add_argument('--patient_thr', type=int, default=100, help='threshold for early stopping')
    parser.add_argument('--data_load_step', type=int, default=3, help='data load step for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch value')
    parser.add_argument('--min_avg_loss', type=float, default=100000.0, help='min avg loss')

    # Map info
    parser.add_argument('--map_size', type=int, default=96, help='width of map image')

    args = parser.parse_args()
    train(args)


def train(args):
    # Print training information
    print_training_info(args)

    # Set device (GPU or CPU)
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = SDDDataset(args, mode='train')
    val_dataset = SDDDataset(args, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = Model(args).to(device)

    # Load pre-trained model if specified
    if args.load_pretrained:
        checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
        model.load_state_dict(torch.load(checkpoint_path))
        print(f">> Loaded model from {checkpoint_path}")

    # Define optimizer
    optimizer_pose = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer_reward = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_epoch = 0
    patient = 0
    min_avg_loss = args.min_avg_loss

    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training phase
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}'):
            xo, xp, xoo, xpo, did = batch
            mo = make_map_batch(xo, did, train_dataset.map, args.map_size).to(device)
            xoo_policy = model.sample(xoo.to(device), mo, model.init_state_enc)
            mo_policy = make_map_batch_for_policy(xo, xoo, xoo_policy, did, train_dataset.map, args.map_size).to(device)

            # Train RNN
            optimizer_pose.zero_grad()
            loss_pose = model(xoo.to(device), mo, mo_policy, xpo.to(device))
            loss_pose.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer_pose.step()

            # Train reward
            optimizer_reward.zero_grad()
            loss_reward = model.cost_reward
            loss_reward.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer_reward.step()

            epoch_loss += loss_pose.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xo, xp, xoo, xpo, did = batch
                mo = make_map_batch(xo, did, val_dataset.map, args.map_size).to(device)
                loss = model(xoo.to(device), mo, None, xpo.to(device))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader)}, Val Loss: {avg_val_loss}')

        # Save model if validation loss improves
        if avg_val_loss < min_avg_loss:
            min_avg_loss = avg_val_loss
            best_epoch = epoch
            patient = 0
            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f">> Model saved to {checkpoint_path}")
        else:
            patient += 1

        # Early stopping
        if patient > args.patient_thr:
            print(f">> Early stopping triggered at epoch {epoch + 1}")
            break


if __name__ == '__main__':
    main()