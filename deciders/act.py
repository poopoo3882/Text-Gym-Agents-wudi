# This file contains functions for interacting with the ChatGPT model

import openai
from .gpt import gpt 
from loguru import logger
from .parser import DISPARSERS, CONPARSERS
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import QianfanChatEndpoint
from memory.env_history import EnvironmentHistory
import tiktoken
import json
import re
import pandas as pd
import pickle
from openai import OpenAI
from .utils import run_chain, get_chat, num_tokens_from_string
from gym.spaces import Discrete


class RandomAct():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state_description, action_description, env_info, game_description=None, goal_description=None):
        if isinstance(self.action_space, Discrete):
            action = self.action_space.sample()+1
        else:
            action = self.action_space.sample()
        return action, '', '', '', 0, 0
    
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
            "player": "item",  # 玩家
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

class NaiveAct(gpt):
    def __init__(self, action_space, args, prompts, distiller, temperature=0.0, max_tokens=2048, logger=None):
        self.action_space = action_space
        self.temperature = temperature
        self.action_desc_dict = args.action_desc_dict
        self.args = args
        self.seed = args.seed 
        self.prompts = prompts
        self.max_generate_tokens = args.max_generate_tokens
        self.cum_token_usage = 0
        self.cum_cost_usage = 0
        self.prompt_level = args.prompt_level
        # load manual 
        manual_data = pd.read_csv('../manual/game_manual_new.csv', encoding='Windows-1252')
        self.game_manual = manual_data.loc[manual_data['Name'] == args.manual_name, 'Website'].values[0]

        # load language traj
        with open('../language_traj/' + args.traj_path, 'rb') as f:
            self.language_traj_list = pickle.load(f)

        if args.gpt_version == "gpt-35-turbo":
            self.model = "gpt-3.5-turbo"
        else:
            self.model = args.gpt_version
        if args.api_type in ["gemma", 'vllm', 'qwen','aistudio', 'groq', 'nvidia']:
            self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        else:
            self.encoding = tiktoken.encoding_for_model(self.model)
        super().__init__(args)
        if self.args.api_type == "azure":
            self.chat = AzureChatOpenAI(
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                azure_endpoint=openai.azure_endpoint,
                openai_api_key=openai.api_key,
                deployment_name=self.args.gpt_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=True,
            )
            from openai import AzureOpenAI
            self.client =  AzureOpenAI(
                api_key=openai.api_key,
                api_version=openai.api_version,
                azure_endpoint=openai.azure_endpoint,
            )
        elif self.args.api_type == "openai":
            self.chat = ChatOpenAI(temperature=self.temperature, base_url=openai.base_url, openai_api_key=openai.api_key, model=self.args.gpt_version)
            self.client =  OpenAI(
                api_key=openai.api_key,
                base_url=openai.base_url,
            )
        elif self.args.api_type == "nvidia":
            self.client = OpenAI(
                api_key=openai.api_key,
                base_url=openai.api_base,
            )
        elif self.args.api_type == "vllm":
            if self.model == 'Meta-Llama-3-8B-Instruct':
                stop_token = "<|eot_id|>"
            else:
                stop_token = "<|im_end|>"
            self.chat = ChatOpenAI(
                openai_api_key='EMPTY',
                base_url=f'http://localhost:{self.args.port}/v1',
                model_name=self.model,
                model_kwargs={"stop": [stop_token]}
            )
            self.client = OpenAI(
                api_key='EMPTY',
                base_url=f'http://localhost:{self.args.port}/v1',
            )
        elif self.args.api_type == 'qwen':
            # self.client = OpenAI(api_key = openai.api_key, base_url = openai.api_base)
            import dashscope 
            dashscope.api_key = openai.api_key
            self.client = dashscope
        elif self.args.api_type == "groq":
            from groq import Groq

            self.client = Groq(
                api_key=openai.api_key,
            )
        elif self.args.api_type == "aistudio":
            import qianfan
            qianfan.get_config().AK = openai.qianfan_ak
            qianfan.get_config().SK = openai.qianfan_sk
            self.client = qianfan.ChatCompletion(model=self.args.gpt_version)
        elif self.args.api_type == "llama":
            # Ollma specifics
            stop_token = "<|im_end|>"
            
            # Set up the chat object using the Ollma local deployment URL
            self.chat = ChatOpenAI(
                openai_api_key='EMPTY',  # Since Ollma is local, API key may not be needed
                base_url=f'http://localhost:{self.args.port}/v1',  # Use the local URL for Ollma
                model_name='llama3.1',
                model_kwargs={"stop": [stop_token]}
            )

            # Set up the client for interacting with Ollma
            self.client = OpenAI(
                api_key='EMPTY',  # Not needed for local Ollma
                base_url=f'http://localhost:{self.args.port}/v1',
            )

        elif self.args.api_type == "qwen7b":
            # Ollma specifics
            stop_token = "<|im_end|>"
            
            # Set up the chat object using the Ollma local deployment URL
            self.chat = ChatOpenAI(
                openai_api_key='EMPTY',  # Since Ollma is local, API key may not be needed
                base_url=f'http://localhost:{self.args.port}/v1',  # Use the local URL for Ollma
                model_name='qwen2.5:7b',
                model_kwargs={"stop": [stop_token]}
            )

            # Set up the client for interacting with Ollma
            self.client = OpenAI(
                api_key='EMPTY',  # Not needed for local Ollma
                base_url=f'http://localhost:{self.args.port}/v1',
            )
        elif self.args.api_type == "gemma":
            # Ollma specifics
            stop_token = "<|im_end|>"
            
            # Set up the chat object using the Ollma local deployment URL
            self.chat = ChatOpenAI(
                openai_api_key='EMPTY',  # Since Ollma is local, API key may not be needed
                base_url=f'http://localhost:{self.args.port}/v1',  # Use the local URL for Ollma
                model_name='gemma:7b',
                model_kwargs={"stop": [stop_token]}
            )

            # Set up the client for interacting with Ollma
            self.client = OpenAI(
                api_key='EMPTY',  # Not needed for local Ollma
                base_url=f'http://localhost:{self.args.port}/v1',
            )


        self.distiller = distiller
        self.fewshot_example_initialization(args.prompt_level, args.prompt_path, distiller = self.distiller)
        if isinstance(self.action_space, Discrete):
            self.default_action = 1
        else:
            self.default_action = [0 for ind in range(self.action_space.shape[0])]
        self.parser = self._parser_initialization()
        self.irr_game_description = ''
        self.memory = []
        self.env_history = EnvironmentHistory()
        self.first_call = True
        self.logger = logger


        if self.prompt_level in [2, 4]: 
            self.memory = self.summarized_fewshot_example
        if args.use_short_mem == 1: 
            self.use_short_mem = True
            self.mem_num = self.args.short_mem_num
        else:
            self.use_short_mem = False
            self.mem_num = 0
        
    def change_key(self,):
        if self.args.api_type == "qwen":
            for key in openai.key_lst:
                if self.client.api_key != key: 
                    self.client.api_key = key

    def update_mem(self,):
        traj = self.game_description 
        traj += self.goal_description
        one_history_token = num_tokens_from_string(self.args.gpt_version, self.env_history.get_one_history())
        history_num = self.args.max_query_tokens // one_history_token
        traj_lst = self.env_history.get_lastest_histories_list(history_num)
        self._update_mem(traj_lst)

    def _update_mem(self, traj_lst):
        my_reflection = self.distiller.generate(self.client, traj_lst, self.memory, self.game_description, self.goal_description, self.action_description)

        self.memory.append(my_reflection)
        self.env_history.reset()


    def clear_mem(self):
        self.update_mem()
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.env_history.reset()

    def reg_parse(self, s):
        if isinstance(self.action_space, Discrete): 
            pattern = r'"action": (\d+)'
        else:
            pattern = r'"action": [-+]?\d*\.\d+'
        match = re.search(pattern, s)
        if match:
            return int(match.group(1))
        else:
            return None

    def _parser_initialization(self):
        if isinstance(self.action_space, Discrete): 
            PARSERS = DISPARSERS
            num_action = self.action_space.n
        else: 
            PARSERS = CONPARSERS
            num_action = self.action_space.shape[0]
            scale = int(self.action_space.high[0])

        if self.args.api_type == "azure":
            autofixing_chat = AzureChatOpenAI(
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                azure_endpoint=openai.azure_endpoint,
                openai_api_key=openai.api_key,
                model=self.args.gpt_version,
                temperature=self.temperature,
                max_tokens=self.max_generate_tokens,
                streaming=True,
                model_kwargs={"seed": self.seed}
            )
        elif self.args.api_type == "openai":
            autofixing_chat = ChatOpenAI(temperature=self.temperature, openai_api_key=openai.api_key,model=self.args.gpt_version, openai_api_base=openai.api_base)
        elif self.args.api_type == "nvidia":
            autofixing_chat = ChatOpenAI(temperature=self.temperature, openai_api_key=openai.api_key, model=self.args.gpt_version, openai_api_base=openai.api_base)
        elif self.args.api_type == "vllm":
            if self.model == 'Meta-Llama-3-8B-Instruct':
                stop_token = "<|eot_id|>"
            else:
                stop_token = "<|im_end|>"
            autofixing_chat = ChatOpenAI(
                openai_api_key='EMPTY',
                base_url=f'http://localhost:{self.args.port}/v1',
                model_name=self.model,
                # stop=["<|eot_id|>"],
                model_kwargs={"stop": [stop_token]},
            )
        elif self.args.api_type == "qwen":
#            breakpoint()
            autofixing_chat = ChatTongyi(dashscope_api_key=openai.api_key, temperature=self.temperature)
        elif self.args.api_type == "groq":
            from langchain_groq import ChatGroq
            autofixing_chat = ChatGroq(groq_api_key=openai.api_key, temperature=self.temperature, model=self.args.gpt_version)
        elif self.args.api_type == "aistudio":
            autofixing_chat = QianfanChatEndpoint(temperature=max(self.temperature, 1e-5), model=self.args.gpt_version, qianfan_ak=openai.qianfan_ak, qianfan_sk=openai.qianfan_sk)
        elif self.args.api_type == "llama":
            autofixing_chat = ChatOpenAI(base_url=f'http://localhost:{self.args.port}/v1',api_key='ollama',model='llama3.1')
        elif self.args.api_type == "qwen7b":
            autofixing_chat = ChatOpenAI(base_url=f'http://localhost:{self.args.port}/v1',api_key='ollama',model='qwen2.5:7b')
        elif self.args.api_type == "gemma":
            autofixing_chat = ChatOpenAI(base_url=f'http://localhost:{self.args.port}/v1',api_key='ollama',model='gemma:7b')
        parser = PydanticOutputParser(pydantic_object=PARSERS[num_action])
        autofixing_parser = OutputFixingParser.from_llm(
            llm=autofixing_chat, parser=parser)
        return autofixing_parser

    def fewshot_example_initialization(self, level, path=None, distiller=None):
        self.fewshot_example = []
        self.irr_few_shot_examples = []
        self.prompt_level = level
        self.expert_knowledge = None
        if level in [1,3,6,7,8]:
            self.irr_few_shot_examples = self.prompts.TASK_IRRELEVANT_PROMPTS
        elif level == 5:
            if hasattr(self.prompts, "expert_prompt"):
                self.expert_knowledge = self.prompts.expert_prompt
            self.fewshot_example = self.prompts.PERCEPTRON_BASIC_FS_EXAMPLES
        else:
            self.irr_few_shot_examples = self.prompts.TASK_IRRELEVANT_PROMPTS
            json_file = f'{path}_l{level}.json'
            with open(json_file, 'r') as infile:
                data = json.load(infile)
            max_step_num = 0
            for traj in data: 
                traj_text = traj[0]['game_description']
                traj_text += traj[0]['goal_description']
                for i, transition in enumerate(traj): 
                    traj_text += transition['observation']
                    traj_text += f"> {transition['action']}"
                    traj_text += f"{transition.get('reward','')}\n"
                    one_traj_token = num_tokens_from_string(traj_text)
                    if one_traj_token > self.args.max_query_tokens:
                        max_step_num = i+1
                        break
                traj_text += f"Your performance is: {transition['cum_reward']}"
            if not max_step_num:
                max_step_num = self.args.max_episode_len
            self.summarized_fewshot_example = self.distiller.generate_from_file(self.client, json_file,max_step_num=max_step_num)

    def response(self, state_description, action_description, env_info, game_description=None, goal_description=None, fewshot_examples=None):
        instruction = "Please suggest an action based on the current game state and the information you get. You must select the appropriate action from the given action descriptions and cannot refrain from taking action or performing any prohibited actions. Your Suggested Action is: "
        messages = []
        messages.append({"role": "system", "content": f"You are an expert-level game player. Your whole response should be in JSON format. You are in a game. {game_description}\n {goal_description}"})
        for my_msg in fewshot_examples:
            messages.append(my_msg)
        messages.append({"role": "user", "content": f"{state_description}.{action_description}\n{instruction}"})

        self.logger.info(f"prompt: {messages}")
        res, usage = get_chat(self.client, messages, api_type=self.args.api_type, model=self.args.gpt_version, temperature=self.temperature, max_tokens=self.max_generate_tokens, seed=self.seed)
        return messages, res, usage
    
    def _add_history_before_action(self, game_description, goal_description, state_description):
        self.game_description = game_description 
        self.goal_description = goal_description
        self.env_history.add("observation", state_description)

        # limit the token used, or it may exceed the max token
        if len(self.env_history):
            one_history_token = num_tokens_from_string(self.args.gpt_version, self.env_history.get_one_history())
            self.env_history.set_history(self.args.max_query_tokens // one_history_token)

    def act(self, state_description, action_description, env_info, game_description=None, goal_description=None, logfile=None):
        self.action_description = action_description
        self._add_history_before_action(game_description, goal_description, state_description)
        asking_round = 0
        res = None
        action = None
        if not self.logger:
            logger.remove()
            self.logger = logger.add(logfile, colorize=True, enqueue=True)
        
        example_messages = []

        # 处理游戏manual
        # prompt level 6: game manual
        if self.args.prompt_level == 6:
            manual_prompt = f"You are in a game. {game_description}\n {goal_description} \n\n This is the game manual for this game. You need to read it carefully and understand the content and play strategies of the game: \n\n\n {self.game_manual}"
            example_messages.append({"role": "system", "name":"example_user",  "content": manual_prompt})
        # 处理RL traj
        # prompt level 7: RL traj
        if self.args.prompt_level == 7:
            formatted_list = [f"[{item}]" for item in self.language_traj_list]
            traj_str =  "\n".join(formatted_list)
            traj_prompt = f"You are in a game. {game_description}\n {goal_description} \n\nThis is the trajectory of playing this game using the RL algorithm. Please read these trajectories carefully and refer to these trajectories to make decisions during the game play:\n\n\n {traj_str} "
            example_messages.append({"role": "system", "name":"example_user",  "content": traj_prompt})
    
        if self.args.prompt_level == 8:
    # 仅对目标和动作描述进行模糊化，而不对游戏描述模糊化
            prompt_processor = ActionProcessor(prompt_level=self.args.prompt_level)
            state_description = prompt_processor.process_prompt(state_description)

        if self.args.prompt_level == 5:    
            if self.fewshot_example:
                for examples in self.fewshot_example:
                    example_messages.append({"role": "system", "name":"example_user",  "content": examples['question']})
                    example_messages.append({"role": "system", "name":"example_assistant",  "content": examples['answer']})
        elif self.args.prompt_level in [2,3,4]:
            if self.prompt_level == 2:
                role_name = "example_user_with_random_policy"
            elif self.prompt_level == 3:
                role_name = "example_user"
            elif self.prompt_level == 4:
                role_name = "example_user_with_expert_policy"
            for mem in self._read_mem():
                example_messages.append({"role": "system", "name": role_name,  "content": mem})
        
        if self.use_short_mem:
            if len(self.env_history) > 1:
                example_messages.append({"role": "user",  "content":  f"{self.env_history.get_histories(self.mem_num)}"})
        
        messages, response, usage = self.response(state_description, action_description, env_info, game_description, goal_description, example_messages)
        
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
        self._add_history_after_action(action)
        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The action is: {action}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')
        token, cost = usage["token"], usage["cost"]
        self.logger.info(f'Token Usage: {token}; Cost Usage: {cost} $.')
        self.cum_token_usage += token
        self.cum_cost_usage += cost
        self.logger.info(f'Cummulative Token Usage: {self.cum_token_usage}; Cummulative Cost Usage: {self.cum_cost_usage} $.')
        return action, messages, response, self.cum_token_usage, self.cum_cost_usage

    def _read_mem(self, ):
        memory = self.memory
        mem_lst = []
        if len(memory) > 5:
            memory = memory[-5:]
        if len(memory) > 0:
            for i, m in enumerate(memory):
                mem_lst.append(f'\nTrial {i}: {m}')
        return mem_lst
        
    def _add_history_after_action(self, action):
        self.env_history.add('action', action)