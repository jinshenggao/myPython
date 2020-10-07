import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
 
# 定义参数
BATCH_SIZE = 32                             # 每一批的训练量
LR = 0.01                                   # 学习率
EPSILON = 0.9                               # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.9                                 # reward discount
TARGET_REPLACE_ITER = 100                   # target的更新频率
MEMORY_CAPACITY = 2000                      # 记忆库大小
env = gym.make('CartPole-v0')               # 立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n              # 杆子能做的动作,假设为2
# print(env.action_space)
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数


ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
 
 
# 创建神经网络模型，输出的是可能的动作
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)	   # N_STATES 输入的节点数量   50 隐层的数量
        self.fc1.weight.data.normal_(0, 0.1)   # 初始化随机权重
        print(self.fc1)
        self.out = nn.Linear(50, N_ACTIONS)    # 50 隐层的数量      N_ACTIONS  输出的节点数量
        self.out.weight.data.normal_(0, 0.1)   # 初始化随机权重
        print(self.out)
 
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
 
 
# 创建Q-learning的模型
class DQN(object):
    def __init__(self):
        # 两张网是一样的，不过就是target_net是每100次更新一次，eval_net每次都更新
        self.eval_net, self.target_net = Net(), Net()
        # DQN需要使用两个神经网络
        # eval为Q估计神经网络 target为Q现实神经网络
 
        self.learn_step_counter = 0                                     # 如果次数到了，更新target_net
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))      # 初始化记忆库用numpy生成一个(2000,6)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式
 
    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # 这里只输入一个 sample,x为场景
        # input only one sample
        if np.random.uniform() < EPSILON:   # 贪婪策略 # 选最优动作
            actions_value = self.eval_net.forward(x)  #将场景输入Q估计神经网络
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # 返回动作最大值
        else:    # 选随机动作
            action = np.random.randint(0, N_ACTIONS) 
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # 比如np.random.randint(0,2)是选择1或0
        print("action:=",action)
        return action
 
    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # 将每个参数打包起来
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
 
    def learn(self):
         # target net 参数更新,每100次
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
             # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
 
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]   #在对应的32个标签位置，返回memory里面的s, [a, r], s_
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])   #取出s
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))  #取出a
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])   #取出r
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])   #取出s_
 
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1) DQL核心公式
        loss = self.loss_func(q_eval, q_target)  #计算误差
 
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()   #反向传递
        self.optimizer.step()




def main():
    dqn = DQN()  # 定义 DQN 系统
 
    print('\nCollecting experience...')
    #  限定训练  400步
    for i_episode in range(40):
        s = env.reset() # 搜集当前环境状态。
        ep_r = 0
        while True:
            env.render()  # 显示实验动画
            a = dqn.choose_action(s)  #选择动作
 
            # 选动作, 得到环境反馈
            s_, r, done, info = env.step(a)
 
            # 修改 reward, 使 DQN 快速学习，看个人需求
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
 
            # 存记忆
            dqn.store_transition(s, a, r, s_)
 
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()   # 记忆库满了就进行学习
                if done:
                    print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
 
            if done:  # 如果回合结束, 进入下回合
                break
            s = s_


if __name__ == '__main__':
    main()