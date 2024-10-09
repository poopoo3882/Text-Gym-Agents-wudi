# 定义手动名称和其他相关参数

game_names = [
        'Asteroids', 'BattleZone', 'Berzerk', 'Bowling', 'Boxing', 'Breakout',
        'DemonAttack', 'Freeway', 'Frostbite', 'Hero', 'MontezumaRevenge',
        'Pitfall','Pong', 'PrivateEye', 'Qbert', 'Riverraid', 'Seaquest',
        'Skiing', 'SpaceInvaders', 'Tennis', 'Venture', 'VideoPinball'
    ]

# 生成命令的函数
def generate_command(decider, prompt_level, num_trails, seed, manual_name, traj_path=None, api_type='qwen7b'):
    command = f"python main_reflexion.py --env_name {env_name} "
    command += f"--init_summarizer {init_summarizer} "
    command += f"--curr_summarizer {curr_summarizer} "
    command += f"--decider {decider} --prompt_level {prompt_level} "
    command += f"--num_trails {num_trails} --seed {seed} "
    command += f"--manual_name {manual_name} --use_short_mem 0 "
    command += f"--max_episode_len 1000 --api_type {api_type}"
    
    if traj_path:
        command += f" --traj_path {traj_path}"
    
    if decider == decider_cot:
        command += " --distiller traj_distiller"
    
    return command


for game in game_names:
    manual_name = game
    env_name = f"Represented{game}-v0"
    init_summarizer = f"Represented{game}_init_translator"
    curr_summarizer = f"Represented{game}_basic_translator"
    traj_path = f"{game}_language_traj_0929.pkl"
    decider_naive = "naive_actor"
    decider_cot = "cot_actor"


    with open(f'/home/wudi/Text-Gym-Agents-wudi/run_shell/qwen/run_{game}_1.sh', 'w') as f:
        # 写入 manual naive 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_naive, 6, 1, seed, manual_name) + '\n')

        # 写入 manual cot 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_cot, 6, 5, seed, manual_name) + '\n')

        # 写入 RL traj naive 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_naive, 7, 1, seed, manual_name, traj_path) + '\n')

        # 写入 RL traj cot 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_cot, 7, 5, seed, manual_name, traj_path) + '\n')
        

    with open(f'/home/wudi/Text-Gym-Agents-wudi/run_shell/qwen/run_{game}_2.sh', 'w') as f:
        # 写入 basic naive 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_naive, 1, 1, seed, manual_name) + '\n')

        # 写入 basic cot 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_cot, 1, 5, seed, manual_name) + '\n')
        
         # 写入 obscure naive 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_naive, 8, 1, seed, manual_name) + '\n')

        # 写入 obscure cot 模式下的命令
        for seed in range(5):
            f.write(generate_command(decider_cot, 8, 5, seed, manual_name) + '\n')

