# MsPacman

## game manual
# python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider naive_actor --prompt_level 6 --num_trails 1  --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --api_type llama

# python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider cot_actor --prompt_level 6 --num_trails 1  --distiller traj_distiller --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --api_type llama

python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider exe_actor --prompt_level 6 --num_trails 3  --distiller guide_generator --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --api_type llama


## RL traj
# python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider naive_actor --prompt_level 7 --num_trails 1  --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --traj_path pong_language_traj_0912.pkl --api_type llama

# python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider cot_actor --prompt_level 7 --num_trails 1  --distiller traj_distiller --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --traj_path MsPacman_language_traj_0929.pkl --api_type llama

python main_reflexion.py --env_name RepresentedMsPacman-v0 --init_summarizer RepresentedMsPacman_init_translator --curr_summarizer RepresentedMsPacman_basic_translator  --decider exe_actor --prompt_level 7 --num_trails 3  --distiller guide_generator --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --traj_path MsPacman_language_traj_0929.pkl --api_type llama
