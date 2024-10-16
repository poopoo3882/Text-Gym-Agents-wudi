import random 
from deciders.utils import get_chat, num_tokens_from_string
import json
from loguru import logger


class TrajPromptSummarizer():
    def __init__(self,args=None,logfile=None):
        self.args = args
        self.seed = args.seed
        with open("/home/wudi/Text-Gym-Agents-master/distillers/traj_summary_few_shot_examples.txt", 'r') as f:
            self.FEW_SHOT_EXAMPLES = f.read()
        
        if logfile:
            # logger.remove()
            logger.add(logfile, colorize=True, enqueue=True, filter=lambda x: '[Reflexion Memory]' in x['message'])

    def generate_from_file(self, client, file_path,max_step_num=200):
        mem = []
        with open(file_path, 'r') as infile:
            data = json.load(infile)

        traj_lst = []
        for traj in data: 
            game_description = traj[0]['game_description']
            goal_description = traj[0]['goal_description']
            action_description = traj[0]['action_description']
            traj_text = ""
            for transition in traj[-max_step_num:]: 
                traj_text += transition['observation']+'\n'
                if type(eval(str(transition['action']))) == type([]):
                    action = float(eval(str(transition['action']))[0])-1
                else:
                    action = transition['action']
                traj_text += f"Action: {action}\n"
                traj_text += f"Reward: {transition['reward']}\n"
                if num_tokens_from_string(self.args.gpt_version, traj_text) > 0.3*self.args.max_query_tokens:
                    traj_lst.append(traj_text)
                    traj_text = ""
            traj_text += f"Your performance is: {transition['cum_reward']}\n"
            traj_lst.append(traj_text)
            reflection = self.generate(client, traj_lst, mem,  game_description, goal_description, action_description, max_len_mem=5)
            mem.append(reflection)
        return mem

    def _generate_summary_query(self, traj_lst, memory, game_description, goal_description, action_description):
        """Allows the Agent to reflect upon a past experience."""
        messages = []
        messages.append({"role": "system",  "content": """You are an analytic and game coach. You need to analyse the game and summarize the current strategy step by step."""})
        # messages.append({"role": "system", "name": "example_assitant", "content": self.FEW_SHOT_EXAMPLES})

        messages.append({"role": "system", "content": f"You are in a game. {game_description} \n {goal_description} \n {action_description}" })
        if len(memory) > 0:
            for i, m in enumerate(memory):
                query = f'Recent Plans. Trial #{i}: {m}\n'
                messages.append({"role": "system", "name": "previous_assistant", "content": query})
        i_traj = 0 
        for traj in traj_lst:
            if i_traj == 0:
                query = f'The current trajectory is: {traj}\n'
            else:
                query = traj
            messages.append({"role": "user", "content": query})
            i_traj += 1
        # truncat messages to make sure the number of tokens of messages are less than self.args.query_token
        instruction_msg = {"role": "user", "content": "Please answer the following questions directly, without additional explanation: 1. Summarize the strategy used in this trajectory and its performance. 2. Summary game-relevant knowledge accordingly that can help others play better. The whole response should be in JSON format with two keys 'Strategy' and 'Knowledge'."}
        messages.append(instruction_msg)
        return messages

    def generate(self, client, traj_lst, memory, game_description, goal_description, action_description, max_len_mem=5):
        if len(memory)> max_len_mem:
            reflection_messages = self._generate_summary_query(traj_lst, memory[-max_len_mem:], game_description, goal_description, action_description)
        else:
            reflection_messages = self._generate_summary_query(traj_lst, memory, game_description, goal_description, action_description)
        reflection, relfexion_usage = get_chat(client, reflection_messages, api_type=self.args.api_type, seed=self.seed )
        logger.info(f'[Traj Summary Memory]The summary prompt is: {reflection_messages}.')
        logger.info(f'[Traj Summary Memory]The summary response is: {reflection}.')
        logger.info(f'[Traj Summary Memory]The summary usage is: {relfexion_usage}.')
        return reflection