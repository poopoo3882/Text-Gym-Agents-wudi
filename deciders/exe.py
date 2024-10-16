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
from langchain.callbacks import FileCallbackHandler
from langchain_community.callbacks import get_openai_callback
from .act import NaiveAct
from memory.env_history import EnvironmentHistory
import tiktoken
from .utils import run_chain
from loguru import logger

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

class EXE(NaiveAct):
    def __init__(self, action_space, args, prompts, distiller, temperature=0., max_tokens=None, logger=None, fixed_suggestion=None, fixed_insight=None):
        super().__init__(action_space, args, prompts, distiller, temperature, max_tokens, logger)
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.num_trails = args.num_trails
        self.game_description = args.game_description
        self.goal_description = args.goal_description
        self.action_description = args.action_description
        self.action_desc_dict = args.action_desc_dict
        self.mem_num = args.short_mem_num
        self.fixed_suggestion = fixed_suggestion
        self.fixed_insight = fixed_insight
        self._update_mem(None)
        self.insight = ""

    def num_tokens_from_string(self,string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens
    
    def update_mem(self,):
        traj = self.game_description 
        traj += self.goal_description
        traj += self.action_description
        traj += str(self.env_history)
        self._update_mem(traj)

    def clear_mem(self):
        self.update_mem()
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.env_history.reset()
        # self._update_mem(None)

    def _update_mem(self, traj):
        if self.memory:
            self.post_memory = self.memory
            self.insight = self.distiller.generate_insight(self.client, self.post_memory)
        else:
            if not self.is_first:
                summary = self.distiller.generate_summary(self.client, traj, self.post_memory)
                self.post_memory.append(summary)
                self.insight = self.distiller.generate_insight(self.client, self.post_memory)
            else:
                self.is_first = False
                self.insight = ""
        suggestion = self.distiller.generate_suggestion(self.client,self.game_description, self.goal_description, self.action_description, self.pre_memory, self.post_memory, self.insight, self.num_trails)
        if self.fixed_suggestion:
            suggestion = self.fixed_suggestion
        if self.fixed_insight:
            self.insight = self.fixed_insight
        self.pre_memory.append(suggestion)
        self.env_history.reset()
        
    def _read_mem(self, ):
        try:
            insight_str = ""
            if self.insight:
                insight_str += "The insights of the game are listed below: "
                insight_str += f"{self.insight}\n"
            suggestion_str = "The suggestions are listed below:" + self.pre_memory[-1][0]
            return insight_str + suggestion_str 
        except:
            return ""
    
    def act(
        self,
        state_description,
        action_description,
        env_info,
        game_description,
        goal_description,
        logfile=None,
    ):
        self.game_description = game_description 
        self.goal_description = goal_description
        self.env_history.add("observation", state_description)

        # print(self.logger)
        reply_format_description = \
            "Your response should choose an optimal action from valid action list, and terminated with following format: "        
            # only task relevant examplesA
        template = "Now you are completing a task."
        template += "You need to carefully understand the description of the game. "

        # TODO: few shot example handle
        if self.irr_few_shot_examples:
            template += "Here are some examples of how you should completing a task."
            for examples in self.irr_few_shot_examples:
                template += "\nQuestion: \n" + examples['question'] + "Answer: \n" + examples['answer']
        
        # add game manual or RL traj
        # prompt level 6: game manual
        if self.args.prompt_level == 6:
            manual_prompt = f"You are in a task. {game_description}\n {goal_description} \n\n This is the game manual for this game. You need to read it carefully and understand the content and play strategies of the game: \n\n\n {self.game_manual}"
            manual_prompt = manual_prompt.replace("{", "{{")
            manual_prompt = manual_prompt.replace("}", "}}")
            template += manual_prompt
        # prompt level 7: RL traj
        elif self.args.prompt_level == 7:
            formatted_list = [f"[{item}]" for item in self.language_traj_list]
            traj_str =  "\n".join(formatted_list)
            traj_prompt = f"You are in a task. {game_description}\n {goal_description} \n\nThis is the trajectory of playing this game using the RL algorithm. Please read these trajectories carefully and refer to these trajectories to make decisions during the game play:\n\n\n {traj_str} "
            traj_prompt = traj_prompt.replace("{", "{{")
            traj_prompt = traj_prompt.replace("}", "}}")
            template += traj_prompt

        else:
            template += "\n\nNow you are in the task.\n" 
            template += f" {game_description}\n{action_description}\n{goal_description}"
            
        if self.args.prompt_level == 8:
            # 仅对目标和动作描述进行模糊化，而不对游戏描述模糊化
            prompt_processor = ActionProcessor(prompt_level=self.args.prompt_level)
            state_description = prompt_processor.process_prompt(state_description)



        template += "You are observing something and  " \
                "you need to choose the optimal action acoordingly."
        template += 'Response and interact using the format: {reply_format_description}{format_instructions}\n'
        
        template += self._read_mem()
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        
        short_memory_template = HumanMessagePromptTemplate.from_template("{history}\nNext is the observation that the agent gets:\n{state_description}Please select an optimal action to gain higher rewards based on the current state and history. The action description is below: {action_description}. Please think step by step. Note: Please Response and interact using the format: {reply_format_description}{format_instructions}\n")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, short_memory_template])
        if self.logger:
            pass
        else:
            if logfile:
                # logger.remove()
                if self.first_call:
                    self.logger = logger.add(logfile, colorize=True, enqueue=True, filter=lambda x: '[Reflexion Memory]' not in x['message'])
                    self.first_call = False
        handler = FileCallbackHandler(logfile)
        total_tokens, total_cost = 0, 0 
        chain = LLMChain(llm=self.chat, prompt=chat_prompt, callbacks=[handler], verbose=False)
        with get_openai_callback() as cb:
            response = run_chain(
                chain,
                game_description=game_description,
                goal_description=goal_description,
                action_description=action_description,
                state_description = self.env_history.get_last_history(),
                history=self.env_history.get_histories(self.mem_num),
                format_instructions=self.parser.get_format_instructions(),
                reply_format_description=reply_format_description,
            )

            total_tokens += cb.total_tokens
            total_cost += cb.total_cost
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
        self.logger.info(f'The optimal action is: {action}.')
        if self.pre_memory:
            self.logger.info(f'The suggestion is: {self.pre_memory[-1]}.')
        if self.post_memory:
            self.logger.info(f'The summary is: {self.post_memory[-1]}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')
        text_prompt = chat_prompt.format_messages(
            game_description=game_description,
            goal_description=goal_description,
            action_description=action_description,
            state_description = self.env_history.get_last_history(),
            history=self.env_history.get_histories(self.mem_num),
            format_instructions=self.parser.get_format_instructions(),
            reply_format_description=reply_format_description,
        )
        text_prompt = f'{text_prompt[0].content}\n{text_prompt[1].content}'
        return action, text_prompt, response, total_tokens, total_cost