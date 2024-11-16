import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from environment_maze import MazeEnv
import cv2
import os
import random
from collections import deque
import gc
import psutil  # 添加psutil库来监控内存使用
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 16
LR = 0.001
GAMMA = 0.99
EPSILON_START = 0.9  # 增加初期探索概率，减少早期频繁撞墙
EPSILON_END = 0.05
EPSILON_DECAY = 0.999  # 减少探索率衰减速度
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000  # 增加经验回放池容量
GRADIENT_ACCUMULATION_STEPS = 4  # 每4个batch更新一次

# DQN network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2)  # 修改输入通道为4，增加访问轨迹通道
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.epsilon = EPSILON_START
        self.learn_step_counter = 0  # 初始化步骤计数器
        self.gradient_accumulation_counter = 0  # 初始化梯度累积计数器

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if random.random() > self.epsilon:  # epsilon-greedy策略
            return self.eval_net(state).argmax().item()
        else:
            return random.randint(0, 3)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.size() < BATCH_SIZE:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        q_next = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        q_target = reward_batch + GAMMA * q_next * (1 - done_batch)

        loss = self.loss_fn(q_eval, q_target)
        if torch.isnan(loss).any():
            print("Loss is NaN, stopping training")
            raise ValueError("Loss is NaN")

        loss.backward()  # 累积梯度，不立即更新

        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)

        # 梯度累积
        if (self.learn_step_counter + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Memory cleanup
        del state_batch, action_batch, reward_batch, next_state_batch, done_batch, q_eval, q_next, q_target, loss
        torch.cuda.empty_cache()  # Clean unused memory

        # 随机探索率衰减
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_END)

        if self.learn_step_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)  # 保存模型

    def save_goal_model(self, path):
        torch.save(self.eval_net.state_dict(), path)


def main():
    env = MazeEnv()
    dqn = DQN()
    EPISODES = 20000  # 测试回合数量
    total_rewards = []

    max_reward = -float('inf')
    best_episode_images = []

    process = psutil.Process(os.getpid())  # 获取当前进程
    visited_positions = np.zeros((32, 32), dtype=bool)  # 初始化访问标记

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0

        step = 0
        trajectory = []  # 保存每一步的位置
        while True:
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)  # 奖励函数在环境中已经包含了对撞墙的处理
            agent_position_new = env.get_position()

            trajectory.append(agent_position_new)  # 记录智能体的位置

            agent_position = env.get_position()
            assert 0 <= agent_position[0] < env.maze.shape[0], f"Invalid position x: {agent_position[0]}"
            assert 0 <= agent_position[1] < env.maze.shape[1], f"Invalid position y: {agent_position[1]}"
            """
            # 修改奖励函数，增加对未访问过区域和接近目标的奖励
            prev_distance = np.linalg.norm(np.array(agent_position) - np.array(env.goal))
            new_distance = np.linalg.norm(np.array(env.get_position()) - np.array(env.goal))
            if new_distance < prev_distance:
                reward += 75  # 增加接近目标的奖励幅度，进一步激励探索

            if not visited_positions[agent_position[0], agent_position[1]]:
                reward += 20  # 对于未访问过的区域给予额外奖励
                visited_positions[agent_position[0], agent_position[1]] = True
            """
            dqn.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            dqn.learn()
            step += 1

            if done:
                total_rewards.append(episode_reward)

                # 保存最佳模型参数
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    best_episode_images = visited_positions.copy()
                    dqn.save_model("dqnmaze_max_reward_model2.pth")

                if env.is_goal_reached():
                    dqn.max_goal_reward = episode_reward
                    dqn.max_goal_reward_params = dqn.eval_net.state_dict()
                    dqn.save_goal_model("dqnmaze_max_goal_reward_model2.pth")
                gc.collect()  # 垃圾回收调用

                memory_info = process.memory_info()
                print(f"Episode {episode}, total reward{episode_reward},Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                break
        """
        if episode_reward == -52:
            # 可视化智能体的路径
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            img[env.maze == 1] = [0, 0, 255]  # 蓝色表示障碍物
            img[env.goal[0], env.goal[1]] = [0, 255, 0]  # 绿色表示目标位置
            for pos in trajectory:
                img[pos[0], pos[1]] = [255, 0, 0]  # 红色表示智能体的路径

            plt.imshow(img)
            plt.title(f"Episode {episode} Trajectory")
            plt.show()
        """
    print("Average total reward over {} episodes: {:.3f}".format(EPISODES, np.mean(total_rewards)))
    print("trajectory:", best_episode_images)

if __name__ == '__main__':
    main()