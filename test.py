import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from environment_maze import MazeEnv
import cv2
import os
import matplotlib.pyplot as plt
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



class Dqn():
    def __init__(self, model_path):
        self.eval_net = Net()
        self.load_model(model_path)

    def load_model(self, model_path):
        self.eval_net.load_state_dict(torch.load(model_path))
        self.eval_net.eval()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_value = self.eval_net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy()
        return action[0]

def main():
    env = MazeEnv()
    dqn = Dqn("dqnmaze_max_reward_model1.pth")
    EPISODES = 1  # 测试回合数量
    total_rewards = []

    max_reward = -float('inf')
    best_episode_images = []

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0

        trajectory = []  # 保存轨迹点
        step = 0
        while True:
            action = dqn.choose_action(state)
            next_state, reward,visited, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            trajectory.append(env.get_position())

            env.render()

            step += 1

            if done:
                total_rewards.append(episode_reward)
                print("Episode {}, total reward: {:.3f}".format(episode, episode_reward))
                break

        if episode_reward > max_reward:
            max_reward = episode_reward
            best_episode_images = trajectory

    print("Average total reward over {} episodes: {:.3f}".format(EPISODES, np.mean(total_rewards)))
    print("trajectory:", best_episode_images)

    # 绘制路径
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[env.maze == 1] = [0, 0, 255]  # 蓝色表示障碍物
    img[env.start[0], env.start[1]] = [255, 0, 0]  # 红色表示起点
    img[env.goal[0], env.goal[1]] = [0, 255, 0]  # 绿色表示终点

    # 将路径可视化
    plt.imshow(img)
    plt.title("Path Planning of Microrobot")

    # 在起点绘制黄色圆圈，并连接到第一个位置
    start_pos = env.start
    first_pos = best_episode_images[0]

    # 绘制起点的黄色圆圈
    plt.plot(start_pos[1] , start_pos[0], 'yo', markersize=5)
    # 将起点与路径的第一个位置连接
    plt.plot([start_pos[1] , first_pos[1]], [start_pos[0] , first_pos[0]], 'y-')

    # 绘制路径上的每个点，确保黄色实心圆点位于方格中心，线连接每个路径点
    for i, pos in enumerate(best_episode_images):
        plt.plot(pos[1] , pos[0], 'yo', markersize=5)  # 黄色实心圆点在方格中心
        if i > 0:  # 从第二个点开始连接线
            prev_pos = best_episode_images[i - 1]
            plt.plot([prev_pos[1] , pos[1] ], [prev_pos[0] , pos[0]], 'y-')  # 黄色线连接

    plt.show()

if __name__ == '__main__':
    main()
