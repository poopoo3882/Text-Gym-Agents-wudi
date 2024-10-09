import openai
from .misc import history_to_str
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain import LLMChain
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain_community.callbacks import get_openai_callback
from .act import NaiveAct
from .utils import run_chain, get_chat, num_tokens_from_string
from gym.spaces import Discrete

class ActionProcessor:
    def __init__(self, prompt_level):
        self.prompt_level = prompt_level
        # 名词映射为 "item"
        self.noun_to_item_dict = {
           "fruit": "item",  # 水果
            "ghost": "item",  # 幽灵
            "missiles": "item",  # 导弹
            "asteroids": "item",  # 小行星
            "tank": "item",  # 坦克
            "missile": "item",  # 导弹
            "compass": "item",  # 指南针
            "tread": "item",  # 履带
            "crosshairs": "item",  # 十字准线
            "robot": "item",  # 机器人
            "evilOtto": "item",  # 邪恶奥托
            "ball": "item",  # 球
            "pins": "item",  # 球瓶
            "opponent": "item",  # 对手
            "block": "item",  # 方块
            "enemy": "item",  # 敌人
            "car": "item",  # 汽车
            "igloo": "item",  # 冰屋
            "iceflow": "item",  # 冰流
            "bear": "item",  # 熊
            "dynamite": "item",  # 炸药
            "skull": "item",  # 骷髅
            "key": "item",  # 钥匙
            "Sue": "item",  # 幽灵名字
            "Inky": "item",  # 幽灵名字
            "Pinky": "item",  # 幽灵名字
            "Blinky": "item",  # 幽灵名字
            "log": "item",  # 原木
            "scorpion": "item",  # 蝎子
            "rope": "item",  # 绳子
            "dove": "item",  # 鸽子
            "red enemy": "item",  # 红色敌人
            "green enemy": "item",  # 绿色敌人
            "fuel": "item",  # 燃料
            "oxygen meter": "item",  # 氧气表
            "diver": "item",  # 潜水员
            "invaders": "item",  # 入侵者
            "sprite": "item",  # 精灵
            "paddle": "item"  # 球拍
        }

    def process_prompt(self, description):
        # 仅在 prompt level = 8 时进行模糊化
        if self.prompt_level == 8:
            words = description.split()
            # 使用字典进行名词映射
            converted_words = [self.noun_to_item_dict.get(word, word) for word in words]
            return " ".join(converted_words)
        return description
    
class ChainOfThought(NaiveAct):
    def __init__(self, action_space, args, prompts, distiller, temperature=0.1, max_tokens=None, logger=None):
        super().__init__(action_space, args, prompts, distiller, temperature, max_tokens,logger)

    def act(
        self,
        state_description,
        action_description,
        env_info,
        game_description,
        goal_description,
        logfile=None,
    ):
        # self.change_key()
        self.action_description = action_description
        self._add_history_before_action(game_description, goal_description, state_description)
        messages = []
        messages.append({"role": "system", "content": f"You are an expert-level game player. Your whole response should be in JSON format. You must carefully understand the Chain-of-Thought method you will use and apply it to the following task. You are in a game. {game_description}\n {goal_description} " })
        
        # task-irrelevant SystemMessage
        if self.irr_few_shot_examples:
            for i, examples in enumerate(self.irr_few_shot_examples):
                messages.append({"role": "system", "name": "example_user", "content": examples['question']})
                messages.append({"role": "system", "name": "example_assistant", "content": examples['answer']})

        if self.fewshot_example:
            for i, examples in enumerate(self.fewshot_example):
                messages.append({"role": "system", "name": "example_user", "content": examples['question']})
                messages.append({"role": "system", "name": "example_assistant", "content": examples['answer']})


        # 处理游戏manual
        # prompt level 6: game manual
        if self.args.prompt_level == 6:
            manual_prompt = f"You are in a game. {game_description}\n {goal_description} \n\n This is the game manual for this game. You need to read it carefully and understand the content and play strategies of the game: \n\n\n {self.game_manual}"
            messages.append({"role": "system", "name":"example_user",  "content": manual_prompt})
        # 处理RL traj
        # prompt level 7: game manual
        if self.args.prompt_level == 7:
            formatted_list = [f"[{item}]" for item in self.language_traj_list]
            traj_str =  "\n".join(formatted_list)
            traj_prompt = f"You are in a game. {game_description}\n {goal_description} \n\nThis is the trajectory of playing this game using the RL algorithm. Please read these trajectories carefully and refer to these trajectories to make decisions during the game play:\n\n\n {traj_str} "
            messages.append({"role": "system", "name":"example_user",  "content": traj_prompt})

        if self.args.prompt_level == 8:
    # 仅对目标和动作描述进行模糊化，而不对游戏描述模糊化
            prompt_processor = ActionProcessor(prompt_level=self.args.prompt_level)
            state_description = prompt_processor.process_prompt(state_description)


        if self.prompt_level in [2, 3, 4]:
            if self.memory:
                if self.prompt_level == 2:
                    role_name = "example_user_with_random_policy"
                elif self.prompt_level == 3:
                    role_name = "example_user"
                elif self.prompt_level == 4:
                    role_name = "example_user_with_expert_policy"
                for mem in self._read_mem():
                    messages.append({"role": "system", "name": role_name,  "content": mem})

        if self.use_short_mem:
            if len(self.env_history) > 1:
                messages.append({"role": "user",  "content":  f"Here is the last {min(self.mem_num, len(self.env_history))} history you have seen: {self.env_history.get_histories(self.mem_num)}"})
                messages.append({"role": "assistant",  "content":  f"I have memorized it."})

        instruction = f'Currently, {state_description}.{action_description}\n Now select your action. You should first take a deep breath. Then you  should think step by step about the action selection and  lay out your thought process explicitly. After that you should decide an action based on the thought. For the whole response, you should use JSON format with two keys "thought process" and "action"'
        if isinstance(self.action_space, Discrete):
            instruction += " where the action should be a scalar."
        else:
            instruction += "."
        instruction_msg = {"role": "user", "content": instruction}
        for i in range(len(messages)):
            if num_tokens_from_string(self.args.gpt_version, messages[:i]) > self.args.max_query_tokens-num_tokens_from_string(self.args.gpt_version, instruction_msg):
                messages = messages[:i-1]
                break
        messages.append(instruction_msg)
        response, usage = get_chat(self.client, messages, api_type=self.args.api_type, model=self.args.gpt_version, temperature=self.temperature, max_tokens=self.max_generate_tokens, seed=self.seed)
        action_str = response
        print(f'my anwser is {action_str}')
        action = None
        for _ in range(5):
            try:
                action = self.reg_parse(response)
                if action is None:
                    action = self.parser.parse(response).action
                break
            except:
                continue
        if not self.logger:
            logger.remove()
            self.logger = logger.add(logfile, colorize=True, enqueue=True)
        self._add_history_after_action(action)
        self.logger.info(f'The GPT prompt is: {messages}.')
        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The optimal action is: {action}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')
        token, cost = usage["token"], usage["cost"]
        self.logger.info(f'Token Usage: {token}; Cost Usage: {cost} $.')
        self.cum_token_usage += token
        self.cum_cost_usage += cost
        self.logger.info(f'Cummulative Token Usage: {self.cum_token_usage}; Cummulative Cost Usage: {self.cum_cost_usage} $.')

        return action, messages, response, token, cost
