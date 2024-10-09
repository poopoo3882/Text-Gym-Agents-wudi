# 使用一个包含Conda的基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /wenwu

# 安装 SWIG 以支持构建 box2d-py
RUN apt-get update && apt-get install -y swig build-essential --fix-missing

# 复制所有内容到容器
COPY . .

# 复制环境配置文件
COPY environment.yaml ./

# 创建Conda环境并激活
RUN conda env create -f environment.yaml && conda clean --all -y

# 确保脚本可以使用conda环境
SHELL ["/bin/bash", "-c"]

# 激活环境并检查环境
RUN echo "source activate llm-gym" > ~/.bashrc

# 默认使用新环境
ENV PATH=/opt/conda/envs/llm-gym/bin:$PATH

# 安装额外的依赖项（通过pip）
# 安装额外的依赖项（通过pip）
RUN /opt/conda/envs/llm-gym/bin/pip install -e /wenwu/atari-representation-learning


# 暴露必要端口（例如用于API或Web应用）
EXPOSE 8000

# 设置默认环境变量，可以通过docker run -e来覆盖
ENV ENVIRONMENT=RepresentedMsPacman-v0 MODEL_PATH=default INIT_SUMMARIZER=RepresentedMsPacman_init_translator PROMPT_LEVEL=1 AGENT=cot_actor API_TYPE=llama

# 运行Python脚本并传递参数
CMD ["sh", "-c", "python main_reflexion.py --env_name $ENVIRONMENT --init_summarizer $INIT_SUMMARIZER --curr_summarizer RepresentedMsPacman_basic_translator --decider $AGENT --prompt_level $PROMPT_LEVEL --num_trails 5 --distiller traj_distiller --seed 0 --manual_name MsPacman --use_short_mem 0 --max_episode_len 1000 --api_type $API_TYPE"]