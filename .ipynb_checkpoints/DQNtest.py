import gym




""" 立杆子游戏的启动过程  """


env = gym.make('CartPole-v0')               # 立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n              # 杆子能做的动作,假设为2
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数
NV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


def main():
    # dqn = DQN()  # 定义 DQN 系统
 
    print('\nCollecting experience...')
    # #  限定训练  400步
    for i_episode in range(400):
        s = env.reset() # 搜集当前环境状态。
        ep_r = 0
        env.render()  # 显示实验动画
        a = 1
        s_, r, done, info = env.step(a)
        # while True:
        #     env.render()  # 显示实验动画
        #     a = 1
        #     s_, r, done, info = env.step(a)



if __name__ == '__main__':
    main()