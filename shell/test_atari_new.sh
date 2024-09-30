# Pong

## game manual
python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider naive_actor --prompt_level 6 --num_trails 1  --seed 0 --manual_name Pong --use_short_mem 0 --max_episode_len 2

python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider cot_actor --prompt_level 6 --num_trails 1  --distiller traj_distiller --seed 0 --manual_name Pong --use_short_mem 0 --max_episode_len 1000

python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider reflexion_actor --prompt_level 6 --num_trails 2  --distiller reflect_distiller --seed 0 --manual_name Pong --use_short_mem 0 --max_episode_len 2

## RL traj
python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider naive_actor --prompt_level 7 --num_trails 1  --seed 0 --traj_path pong_language_traj_0912.pkl --use_short_mem 0 --max_episode_len 2

python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider cot_actor --prompt_level 7 --num_trails 2  --distiller traj_distiller --seed 0 --traj_path pong_language_traj_0912.pkl --use_short_mem 0 --max_episode_len 2

python main_reflexion.py --env_name RepresentedPong-v0 --init_summarizer RepresentedPong_init_translator --curr_summarizer RepresentedPong_basic_translator  --decider reflexion_actor --prompt_level 7 --num_trails 2  --distiller reflect_distiller --seed 0 --traj_path pong_language_traj_0912.pkl --use_short_mem 0 --max_episode_len 2
