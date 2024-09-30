import gym
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from atariari.benchmark.wrapper import AtariARIWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 1. 定义一个函数来创建包裹的环境
def make_ari_atari_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id)
        env = AtariARIWrapper(env)  # 包裹 AtariARIWrapper
        env.seed(seed)
        return env
    return _init

# 2. 使用 DummyVecEnv 来并行化处理多个包裹的环境
vec_env = DummyVecEnv([make_ari_atari_env("PongNoFrameskip-v4", seed=i) for i in range(4)])

# 3. 使用 FrameStack 包装以便捕捉帧动态
vec_env = VecFrameStack(vec_env, n_stack=4)

# 4. 加载或训练强化学习模型
model = A2C("CnnPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50_000)
