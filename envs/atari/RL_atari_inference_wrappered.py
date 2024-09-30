import gym
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from atariari.benchmark.wrapper import AtariARIWrapper  # 假设你的 wrapper 在 this_module 中

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
vec_env = DummyVecEnv([make_ari_atari_env("PongNoFrameskip-v4", seed=i) for i in range(1)])

# 3. 使用 FrameStack 包装以便捕捉帧动态
vec_env = VecFrameStack(vec_env, n_stack=4)

# 加载保存的模型
model = A2C.load("a2c_pong_model_new")

# 5. 推理和获取标签
obs = vec_env.reset()

max_num = 2000
gap = 20
while max_num:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, infos = vec_env.step(action)
    
    # 从 info 中获取 labels
    if not max_num % gap:
        for info in infos:
            if "labels" in info:
                print('Action:', action[0])
                print("Labels:", info["labels"])
                print('Reward:', rewards[0])

    max_num = max_num - 1
    
    # vec_env.render("human")


    

