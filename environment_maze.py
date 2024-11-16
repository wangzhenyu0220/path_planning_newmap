import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        self.maze = np.zeros((32, 32), dtype=np.uint8)  # 初始化迷宫

        # 设定起始位置和目标位置
        self.start = (6, 24)
        self.goal = (26, 2)
        self.agent_position = self.start

        # 设定观测空间和动作空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 32, 32), dtype=np.uint8)  # 修改为4通道，包含访问轨迹
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.visited = np.zeros((32, 32), dtype=np.uint8) 
        self._generate_maze()

    def _generate_maze(self):
        self.maze[0, :] = 1
        self.maze[31, :] = 1
        self.maze[1:31, 0] = 1
        self.maze[1:31, 31] = 1

        # 在迷宫中设置障碍物
        for i in range(1,19):
            for j in range(1,4):
                self.maze[i,j] = 1

        for i in range(7,10):
            for j in range(4,10):
                self.maze[i, j] = 1

        for i in range(10,13):
            for j in range(4,7):
                self.maze[i, j] = 1

        for i in range(16,22):
            for j in range(4,7):
                self.maze[i, j] = 1

        for i in range(25,28):
            for j in range(4,10):
                self.maze[i, j] = 1

        for i in range(13,16):
            for j in range(10,13):
                self.maze[i, j] = 1

        for i in range(19,22):
            for j in range(10,19):
                self.maze[i, j] = 1

        for i in range(10,16):
            for j in range(16,19):
                self.maze[i, j] = 1

        for i in range(25,31):
            for j in range(16,19):
                self.maze[i, j] = 1

        for i in range(1,7):
            for j in range(13,16):
                self.maze[i, j] = 1

        for i in range(1,4):
            for j in range(16,28):
                self.maze[i, j] = 1

        for i in range(10,13):
            for j in range(22,28):
                self.maze[i, j] = 1

        for i in range(12,28):
            for j in range(22,25):
                self.maze[i, j] = 1

        for i in range(25,28):
            for j in range(25,28):
                self.maze[i, j] = 1

        for i in range(16,22):
            for j in range(28,31):
                self.maze[i, j] = 1

    def reset(self):
        self.agent_position = self.start
        self.visited = np.zeros((32, 32), dtype=np.uint8)  # 记录访问轨迹
        return self._get_observation()

    def step(self, action):
        x, y = self.agent_position
        new_x, new_y = x, y

        if action == 0:  # 上
            new_x -= 1
        elif action == 1:  # 下
            new_x += 1
        elif action == 2:  # 左
            new_y -= 1
        elif action == 3:  # 右
            new_y += 1

        new_x = np.clip(new_x, 0, self.maze.shape[0] - 1)
        new_y = np.clip(new_y, 0, self.maze.shape[1] - 1)

        reward = 0
        done = False

        if self.maze[new_x, new_y] == 1:
            reward = -10  # 增加撞墙惩罚
            done = True
        else:
            prev_distance = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal))
            self.agent_position = (new_x, new_y)
            new_distance = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal))

            # 更新访问轨迹
            if not self.visited[new_x, new_y]:
                exploration_reward = max(2, 10 - 0.3 * np.sum(self.visited))
                reward += exploration_reward  # 对于未访问过的区域给予额外奖励

            self.visited[new_x, new_y] = 1

            # 奖励函数：接近目标的奖励
            if new_distance < prev_distance:
                reward += 20  # 接近目标的奖励

            done = self.agent_position == self.goal
            if done:
                reward += 500  # 到达目标的额外奖励
            else:
                reward -= 1  # 每步负奖励以鼓励快速到达目标

        return self._get_observation(), reward, self.visited,done, {}

    def _get_observation(self):
        obs = np.zeros((4, 32, 32), dtype=np.uint8)
        obs[0, self.agent_position[0], self.agent_position[1]] = 1
        obs[1, self.goal[0], self.goal[1]] = 1
        obs[2, self.maze == 1] = 1
        obs[3] = self.visited  # 添加访问轨迹通道
        return obs

    def render(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[self.agent_position[0], self.agent_position[1]] = [255, 0, 0]  # 红色表示智能体位置
        img[self.goal[0], self.goal[1]] = [0, 255, 0]  # 绿色表示目标位置
        img[self.maze == 1] = [0, 0, 255]  # 蓝色表示障碍物
        plt.imshow(img)
        plt.grid(False)
        plt.show()

    def get_image(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[self.agent_position[0], self.agent_position[1]] = [255, 0, 0]  # Red for the agent
        img[self.goal[0], self.goal[1]] = [0, 255, 0]  # Green for the goal
        img[self.maze == 1] = [0, 0, 255]  # Blue for walls
        return img

    def get_position(self):
        # 返回智能体当前位置
        return self.agent_position

    def is_goal_reached(self):
        # 检查智能体是否到达目标点
        return self.agent_position == self.goal


if __name__ == '__main__':
    env = MazeEnv()
    observation = env.reset()
    env.render()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
