import random 
from deciders.utils import get_chat
import json
from loguru import logger


class TrajPromptSummarizer():
    def __init__(self,args=None,logfile=None):
        self.args = args
        self.seed = args.seed
        with open("./distillers/traj_summary_few_shot_examples.txt", 'r') as f:
            self.FEW_SHOT_EXAMPLES = f.read()
        
        if logfile:
            # logger.remove()
            logger.add(logfile, colorize=True, enqueue=True, filter=lambda x: '[Reflexion Memory]' in x['message'])

    def generate_from_file(self, file_path,max_step_num=200):
        mem = []
        with open(file_path, 'r') as infile:
            data = json.load(infile)
        for traj in data: 
            traj_text = traj[0]['game_description']+'\n'
            traj_text += traj[0]['goal_description']+'\n'
            for transition in traj[-max_step_num:]: 
                traj_text += transition['observation']+'\n'
                if type(eval(str(transition['action']))) == type([]):
                    action = float(eval(str(transition['action']))[0])-1
                else:
                    action = transition['action']
                traj_text += f"Action: {action}\n"
                traj_text += f"Reward: {transition['reward']}\n"
            traj_text += f"Your performance is: {transition['cum_reward']}\n"
            reflection = self.generate(traj_text, mem, max_len_mem=5)
            mem.append(reflection)        
        return mem

    def _generate_summary_query(self, traj, memory):
        """Allows the Agent to reflect upon a past experience."""
        messages = []
        messages.append({"role": "system",  "content": """You will be given the history of a past experience in which you were placed in an environment and given a task to complete. Summarize your trajectory and reasoning the relation between your policy and the obtained result."""})
        messages.append({"role": "system", "name": "example_assitant", "content": self.FEW_SHOT_EXAMPLES})

        query = traj
        if len(memory) > 0:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(memory):
                query += f'Trial #{i}: {m}\n'
        query += '\n\nPlease give your new plan.'
        messages.append({"role": "user", "content": self.FEW_SHOT_EXAMPLES})
        return messages

    def generate(self, traj, memory, max_len_mem=5):
        if len(memory)> max_len_mem:
            reflection_messages = self._generate_summary_query(traj, memory[-max_len_mem:])
        else:
            reflection_messages = self._generate_summary_query(traj, memory)
        reflection = get_chat(reflection_messages, api_type=self.args.api_type, model=self.args.gpt_version, seed=self.seed)
        logger.info(f'[Reflexion Memory]The reflexion prompt is: {reflection_messages}.')
        logger.info(f'[Reflexion Memory]The reflexion response is: {reflection}.')
        return reflection
