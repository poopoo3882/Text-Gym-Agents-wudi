import argparse
import envs
import deciders
import distillers
from matplotlib import animation
import matplotlib.pyplot as plt
import prompts as task_prompts
import os
import datetime
import time
from collections import deque
from envs.translator import InitSummarizer, CurrSummarizer, FutureSummarizer, Translator
import gym
import json
import pandas as pd
import random
import numpy as np
import datetime
from loguru import logger
from gym.spaces import Discrete
import gym
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from atariari.benchmark.wrapper import AtariARIWrapper  # 假设你的 wrapper 在 this_module 中

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(
        description="use a RL model to generate expert knowledge"
    )

parser.add_argument(
        "--init_summarizer",
        type=str,
        default='RepresentedPong_init_translator',
        help="The name of the init summarizer to use.",
    )

parser.add_argument(
    "--curr_summarizer",
    type=str,
    default='RepresentedPong_basic_translator',
    help="The name of the curr summarizer to use.",
)

parser.add_argument(
    "--env",
    type=str,
    default="base_env",
    help="The name of the gym environment to use.",
)

parser.add_argument(
    "--env_name",
    type=str,
    default="RepresentedPong-v0",
    help="The name of the gym environment to use.",
)

parser.add_argument(
        "--is_only_local_obs",
        type=int,
        default=1,
        help="Whether only taking local observations, if is_only_local_obs = 1, only using local obs"
    )

parser.add_argument(
        "--max_episode_len",
        type=int,
        default=108000//6,
        help="The maximum number of steps in an episode",
    )

parser.add_argument(
        "--rl_env_name",
        type=str,
        default= "PongNoFrameskip-v4",
    )

parser.add_argument(
        "--rl_model_name",
        type=str,
        default= "a2c_pong_model_new",
    )

# 建立translator
args = parser.parse_args()
env_class = envs.REGISTRY[args.env]
init_summarizer = InitSummarizer(envs.REGISTRY[args.init_summarizer], args)
curr_summarizer = CurrSummarizer(envs.REGISTRY[args.curr_summarizer])
sampling_env = envs.REGISTRY["sampling_wrapper"](gym.make(args.env_name))
translator = Translator(
        init_summarizer, curr_summarizer, None, env=sampling_env
    )

# 运行RL算法，翻译轨迹，得到结果
def make_ari_atari_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id)
        env = AtariARIWrapper(env)  # 包裹 AtariARIWrapper
        env.seed(seed)
        return env
    return _init

vec_env = DummyVecEnv([make_ari_atari_env(args.rl_env_name, seed=i) for i in range(1)])
vec_env = VecFrameStack(vec_env, n_stack=4)
model = A2C.load(args.rl_model_name)
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
                print('Action:', action)
                print("Labels:", info["labels"])
                
                print('Reward:', rewards)

    max_num = max_num - 1
    
    # vec_env.render("human")
