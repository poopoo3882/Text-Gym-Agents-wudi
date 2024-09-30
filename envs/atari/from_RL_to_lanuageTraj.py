import gym
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from atariari.benchmark.wrapper import AtariARIWrapper  
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def get_traj(task, action, label, reward):
    traj = ""
    if task == 'pong':
        if action == 1:
            action_desc = f"'Do nothing'"
        elif action == 2:
            action_desc = f"'Hit your ball'"
        elif action == 3:
            action_desc = f"'Move right'"
        elif action == 4:
            action_desc = f"'Move left'"
        elif action == 5:
            action_desc = f"'Move right while hiting the ball'"
        else:
            action_desc = f"'Move left while hiting the ball'"
        traj += f"You are at position ({label['player_y'], label['player_x']}, your opponent is at position ({label['enemy_y'], label['enemy_x']}) ), the ball is at ({label['ball_y'], label['ball_x']})" \
               f"your oppoent's score is {label['enemy_score']}, your score is {label['player_score']}."
        
        traj += f"\n You took action {action}:{action_desc}"

        traj += f"\n You obtained reward after taking the action: You get rewards {reward}"
    
    return traj



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
language_traj_list = []

# RL推理
while max_num:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, infos = vec_env.step(action)
    
    # 从 info 中获取 labels，生成language traj
    if not max_num % gap:
        for info in infos:
            if "labels" in info:
                language_traj_list.append(get_traj(task='pong',action=action[0],label=info["labels"],reward=rewards[0]))


    max_num = max_num - 1
    

# 保存language traj
print(len(language_traj_list))
print(language_traj_list[0])
with open('pong_language_traj_0912.pkl', 'wb') as file:
        pickle.dump(language_traj_list, file)

    

