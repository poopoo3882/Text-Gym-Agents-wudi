from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# 使用 GPU 5
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 创建 Atari 环境（Pong）
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
# 使用 4 帧堆叠
vec_env = VecFrameStack(vec_env, n_stack=4)

# 加载保存的模型
model = A2C.load("a2c_pong_model")

# 重置环境
obs = vec_env.reset()

# 进入推理循环，使用模型进行游戏并渲染
num = 10000
while num:
    # 预测动作
    action, _states = model.predict(obs, deterministic=False)
    # 执行动作
    obs, rewards, dones, info = vec_env.step(action)
    # 渲染游戏界面
    vec_env.render("human")

    num = num - 1