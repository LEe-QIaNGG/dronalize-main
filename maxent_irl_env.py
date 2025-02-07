
# Simulate trajectory
for i in vehicles:
    # select vehicle
    vehicle_id = i
    print('Target Vehicle: {}'.format(vehicle_id))

    # create environment
    env = NGSIMEnv(scene='us-101', period=period, vehicle_id=vehicle_id, IDM=False)

    # Data collection
    length = env.vehicle.ngsim_traj.shape[0]
    timesteps = np.linspace(10, length-60, num=50, dtype=np.int16)
    train_steps = np.random.choice(timesteps, size=10, replace=False)

    # run until the road ends
    for start in train_steps:
        # go the the scene
        env.reset(reset_time=start)

        # set up buffer of the scene
        buffer_scene = []

        # target sampling space
        lateral_offsets, target_speeds = env.sampling_space()
            
        # trajectory sampling
        buffer_scene = []
        print('Timestep: {}'.format(start))

        for lateral in lateral_offsets:
            for target_speed in target_speeds:
                # sample a trajectory
                action = (lateral, target_speed, 5)
                obs, features, terminated, info = env.step(action)

                # render env
                if render_env:
                    env.render()

                # get the features
                traj_features = features[:-1]
                human_likeness = features[-1]
                
                # add scene trajectories to buffer
                buffer_scene.append((lateral, target_speed, traj_features, human_likeness))

                # set back to previous step
                env.reset(reset_time=start)
        
        # calculate human trajectory feature
        env.reset(reset_time=start, human=True)
        obs, features, terminated, info = env.step()

        # eliminate invalid examples
        if terminated or features[-1] > 2.5:
            continue

        # process data            
        human_traj = features[:-1]
        buffer_scene.append([0, 0, features[:-1], features[-1]])

        # add to buffer
        human_traj = buffer_scene[-1][2]
        human_traj_features.append(human_traj)
        buffer.append(buffer_scene)

# normalize features
max_v = np.max([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)
min_v = np.min([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)
max_v[6] = 1.0

for scene in buffer:
    for traj in scene:
        for i in range(feature_num):
            traj[2][i] /= max_v[i]
'''
buffer = [
    scene_1,  # 一个场景中的轨迹列表
    scene_2,
    ...
]

scene = [
    (lateral_1, target_speed_1, traj_features_1, human_likeness_1),
    (lateral_2, target_speed_2, traj_features_2, human_likeness_2),
    ...
    (0, 0, human_traj_features, human_likeness_human),  # 专家轨迹
]
lateral：轨迹的横向偏移量（例如，在车道内或跨车道的横向动作）。
target_speed：轨迹的目标速度（纵向速度的控制）。
traj_features：轨迹的特征向量，描述该轨迹的状态和环境特征。
这可能包括速度、加速度、横向偏移、目标点等信息。
human_likeness：该轨迹与人类驾驶轨迹的相似性分数（通常由 features[-1] 提供）。
此外，每个场景还包括一个人类驾驶轨迹（即实际由专家生成的轨迹，标注为人类行为）。
'''


'''#### MaxEnt IRL ####'''    
print("Start training...")
# initialize weights
# 初始化神经网络和优化器
reward_net = IRLRewardModel(input_dim=feature_num)  # 输入特征维度
optimizer = optim.Adam(reward_net.parameters(), lr=0.001)  # 使用Adam优化器
lam = 0.01  # 正则化系数

# 训练循环
for iteration in range(n_iters):
    print('iteration: {}/{}'.format(iteration + 1, n_iters))

    feature_exp = np.zeros([feature_num])  # 模型的特征期望
    human_feature_exp = np.zeros([feature_num])  # 人类轨迹特征期望
    index = 0
    log_like_list = []
    iteration_human_likeness = []
    num_traj = 0

    for scene in buffer:
        # 转换成可处理的scene trajs，加上计算奖励值
        scene_trajs = []
        for trajectory in scene:
            features = torch.tensor(trajectory[2], dtype=torch.float32)
            reward = reward_net(features).item()  # 用神经网络计算奖励值
            scene_trajs.append((reward, trajectory[2], trajectory[3]))  # 奖励值, 特征, 人类相似性

        # 计算轨迹概率分布
        rewards = [traj[0] for traj in scene_trajs]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        probs = torch.softmax(rewards, dim=0).numpy()  # 轨迹概率

        # 计算特征期望（模型预测）
        traj_features = np.array([traj[1] for traj in scene_trajs])
        feature_exp += np.dot(probs, traj_features)  # 加权平均计算特征期望

        # 计算对数似然
        log_like = np.log(probs[-1] / np.sum(probs))
        log_like_list.append(log_like)

        # 选择轨迹计算人类相似性
        idx = probs.argsort()[-3:][::-1]  # 概率最高的三个轨迹
        iteration_human_likeness.append(np.min([scene_trajs[i][-1] for i in idx]))

        # 计算人类轨迹的特征期望
        human_feature_exp += human_traj_features[index]

        num_traj += 1
        index += 1

    # 计算梯度损失（负对数似然 + 正则化项）
    feature_exp = torch.tensor(feature_exp / num_traj, dtype=torch.float32)
    human_feature_exp = torch.tensor(human_feature_exp / num_traj, dtype=torch.float32)
    loss = torch.norm(human_feature_exp - feature_exp) + lam * torch.sum(torch.square(torch.cat([p.view(-1) for p in reward_net.parameters()])))
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录训练日志
    with open('deep_training_log.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([iteration + 1, human_feature_exp.detach().numpy(), feature_exp.detach().numpy(),
                            np.linalg.norm(human_feature_exp.detach().numpy() - feature_exp.detach().numpy()),
                            loss.item(), iteration_human_likeness, np.mean(iteration_human_likeness),
                            np.sum(log_like_list) / num_traj])

#### run test ###
for id in vehicles:
    # select vehicle
    vehicle_id = id
    print('Target Vehicle: {}'.format(vehicle_id))

    # create training log
    with open('general_testing_log_{}.csv'.format(vehicle_id), 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(['vehicle', 'scene', 'human likeness', 'weights', 'max features', 'min features', 'FDE']) 

    # create environment
    env = NGSIMEnv(scene='us-101', period=period, vehicle_id=vehicle_id, IDM=False)

    # Data collection
    length = env.vehicle.ngsim_traj.shape[0]
    timesteps = np.linspace(10, length-60, num=50, dtype=np.int16)
    test_steps = np.random.choice(timesteps, size=10, replace=False)

    # begin planning
    for start in test_steps:
        # go to the scene
        env.reset(reset_time=start)

        # determine target sampling space
        lateral_offsets, target_speeds = env.sampling_space()
                
        # set up buffer of the scene
        buffer_scene = []

        # lateral and speed trajectory sampling
        print('scene: {}, sampling...'.format(start))
        for lateral in lateral_offsets:
            for target_speed in target_speeds:
                # sample a trajectory
                action = (lateral, target_speed, 5)
                obs, features, terminated, info = env.step(action)

                # render env
                if render_env:
                    env.render()
                        
                # get the features
                traj_features = features[:-1]
                human_likeness = features[-1]

                # add the trajectory to scene buffer
                buffer_scene.append([lateral, target_speed, traj_features, human_likeness])

                # go back to original scene
                env.reset(reset_time=start)

        # normalize features
        for traj in buffer_scene:
            for i in range(feature_num):
                if max_v[i] == 0:
                    traj[2][i] = 0
                else:
                    traj[2][i] /= max_v[i] 
        
        # evaluate trajectories
        reward_HL = []
        for trajectory in buffer_scene:
            reward = np.dot(trajectory[2], theta)
            reward_HL.append([reward, trajectory[3]]) # reward, human likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in reward_HL]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # select trajectories to calculate human likeness
        idx = probs.argsort()[-3:][::-1]
        HL = np.min([reward_HL[i][-1] for i in idx])

        # add to testing log
        with open('general_testing_log_{}.csv'.format(vehicle_id), 'a') as csvfile:  
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow([vehicle_id, start, HL, theta, max_v, min_v, [reward_HL[i][-1] for i in range(len(reward_HL))]])