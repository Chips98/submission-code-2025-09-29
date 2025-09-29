# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E402
from __future__ import annotations  # 导入未来版本的注解功能，允许在类型提示中使用尚未定义的类

import argparse  # 导入命令行参数解析模块
import asyncio  # 导入异步IO模块，用于处理异步编程
import json  # 导入JSON处理模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块
import random  # 导入随机数生成模块
import sys  # 导入系统特定参数和函数模块
import time  # 导入时间模块，用于生成随机种子
import warnings  # 导入警告模块，用于控制警告信息
from datetime import datetime, timedelta  # 从datetime模块导入日期时间和时间差类
from typing import Any  # 从typing模块导入Any类型，用于类型提示
import sqlite3  # 导入sqlite3模块，用于访问数据库
import pdb
from colorama import Back, Fore, Style # 从colorama导入Back，用于控制台文本背景色
from yaml import safe_load  # 从yaml导入safe_load函数，用于安全加载YAML文件
from tqdm import tqdm  # 导入tqdm用于显示进度条
from utils import *


# 禁用异步警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")
logging.getLogger("asyncio").setLevel(logging.ERROR)

# 将项目根目录添加到Python路径中，以便导入项目模块
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 额外确保当前目录也在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入项目中的各个模块
from asce.clock.clock import Clock  # 导入时钟模块
from asce.social_agent.agents_generator import (gen_control_agents_with_data,
                                                 generate_reddit_agents)  # 导入代理生成器
from asce.social_platform.channel import Channel  # 导入通信通道模块
from asce.social_platform.platform import Platform  # 导入平台模块
from asce.social_platform.typing import ActionType  # 导入动作类型枚举

# 导入认知引导引擎
try:
    from guidance.core.main_controller import CGEMainController
    CGE_AVAILABLE = True
    social_log.info("认知引导引擎(CGE)模块导入成功")
except ImportError as e:
    CGE_AVAILABLE = False
    social_log.warning(f"认知引导引擎(CGE)模块导入失败: {e}")

# 导入自定义工具函数
from utils import (
    AsyncTqdmWrapper,
    generate_hybrid_user_profiles,
    export_data,
    load_user_contexts,
    export_user_context_data,
    load_cognition_space,
    get_individual_simulation_progress
)

# 使用新的统一日志系统
from asce.social_agent.logging_system import get_logging_system
social_log = None
platform_log = None
agent_log = None
comprehensive_log = None

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Arguments for script.")  # 创建参数解析器对象
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)  # 添加配置文件路径参数
parser.add_argument(
    "--seed",
    type=int,
    help="Random seed for reproducibility. If not provided, a random seed will be generated.",
    required=False,
    default=None,
)  # 添加随机种子参数




# 移到模块化函数中配置
asyncio.get_event_loop().set_debug(False)


# ========== 模块化函数 ==========

def setup_logging():
    """
    设置统一日志系统
    使用新的ASCE日志系统，在 /log 目录下创建以时间命名的子目录，
    包含四个核心日志文件：agent、platform、simulation、config
    
    返回:
        tuple: (log_dir, platform_log, agent_log, simulation_log)
    """
    global social_log, platform_log, agent_log, comprehensive_log
    
    # 获取统一的日志系统
    logging_system = get_logging_system()
    log_dir = logging_system.setup_session_logging()
    
    # 获取各种日志器
    platform_log = logging_system.get_platform_logger()
    agent_log = logging_system.get_agent_logger()
    simulation_log = logging_system.get_simulation_logger()
    
    # 向后兼容
    social_log = simulation_log
    comprehensive_log = simulation_log
    
    # 配置库日志输出，禁止传播到控制台
    for logger_name in ["asce", "camel", "social.agent", "social.twitter"]:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
    
    # 清理旧日志文件
    logging_system.cleanup_old_logs()
    
    simulation_log.info(f"统一日志系统初始化完成，日志目录: {log_dir}")
    
    return log_dir, platform_log, agent_log, simulation_log


def load_and_save_config(config_path):
    """
    读取配置文件并使用统一日志系统保存配置信息
    
    参数:
        config_path: 配置文件路径
        
    返回:
        dict: 配置参数字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = safe_load(f)
    
    # 使用统一日志系统保存配置
    logging_system = get_logging_system()
    
    # 准备配置数据
    config_data = {
        "配置文件路径": config_path,
        "配置加载时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 添加各个配置节
    for section_name, section_data in cfg.items():
        if isinstance(section_data, dict):
            config_data[f"--- {section_name.upper()} ---"] = ""
            for key, value in section_data.items():
                config_data[key] = value
        else:
            config_data[section_name] = section_data
    
    # 保存配置参数
    logging_system.save_config_parameters(config_data)
    
    return cfg


def setup_random_seed(args, config_path):
    """
    设置随机种子
    
    参数:
        args: 命令行参数
        config_path: 配置文件路径
        
    返回:
        int: 设置的随机种子值
    """
    if args.seed is not None:
        # 使用命令行参数提供的种子
        random_seed = args.seed
    else:
        # 检查配置文件中是否有随机种子设置
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_temp = safe_load(f)
                simulation_params_temp = cfg_temp.get("simulation", {})
                config_seed = simulation_params_temp.get("random_seed")
                if config_seed is not None:
                    random_seed = config_seed
                else:
                    # 使用当前时间生成随机种子
                    random_seed = int(time.time()) % 10000
        else:
            # 使用当前时间生成随机种子
            random_seed = int(time.time()) % 10000
    
    # 设置Python的random模块种子
    random.seed(random_seed)
    
    # 如果使用了numpy，也设置numpy的随机种子
    try:
        import numpy as np
        np.random.seed(random_seed)
        social_log.info("NumPy随机种子已设置")
    except ImportError:
        social_log.info("NumPy不可用，仅设置Python随机种子")
    
    # 记录随机种子到日志
    print(f"{Fore.YELLOW}使用随机种子: {random_seed}{Fore.RESET}")
    social_log.info(f"使用随机种子: {random_seed}")
    
    return random_seed


def setup_cge_engine(guidance_engine_config, guidance_tasks_config, inference_configs, infra, db_path):
    """
    设置认知引导引擎（CGE）
    
    参数:
        guidance_engine_config: 引导引擎配置
        guidance_tasks_config: 引导任务配置
        inference_configs: 推理配置
        infra: 平台对象
        db_path: 数据库路径
        
    返回:
        CGEMainController: CGE控制器对象，如果失败则返回None
    """
    cge_controller = None
    
    if CGE_AVAILABLE and guidance_engine_config and guidance_engine_config.get("enabled", False):
        try:
            # 构建完整配置
            full_config = {
                'guidance_engine': guidance_engine_config,
                'guidance_tasks': guidance_tasks_config or [],
                'inference': inference_configs
            }
            
            cge_controller = CGEMainController(full_config, infra, db_path)
            social_log.info(f"认知引导引擎初始化成功，配置了{len(guidance_tasks_config)}个引导任务")
            
        except Exception as e:
            social_log.error(f"认知引导引擎初始化失败: {e}")
            cge_controller = None
    elif guidance_engine_config and guidance_engine_config.get("enabled", False):
        social_log.warning("引导引擎已启用但CGE模块不可用，跳过引导功能")
    
    return cge_controller


def initialize_user_data(params):
    """
    初始化用户数据
    
    参数:
        params: 参数字典
        
    返回:
        str: 用户数据文件路径
    """
    if params['real_user_ratio'] == 1:
        user_path = params['real_user_path']
    else:
        # 检查这个路径是否存在
        if not os.path.exists(os.path.dirname(params['hybrid_user_profiles_path'])):
            os.makedirs(os.path.dirname(params['hybrid_user_profiles_path']), exist_ok=True)
            social_log.info(f"生成新的混合用户文件夹: {os.path.dirname(params['hybrid_user_profiles_path'])}")
        if os.path.exists(params['hybrid_user_profiles_path']):
            user_path = params['hybrid_user_profiles_path']
            social_log.info(f"使用已存在的混合用户文件: {user_path}")
        else:
            user_path = generate_hybrid_user_profiles(
                params['real_user_ratio'], 
                params['num_agents'], 
                params['real_user_path'], 
                params['random_user_path'], 
                params['hybrid_user_profiles_path']
            )
            social_log.info(f"生成新的混合用户文件: {user_path}")
    
    # 检查混合用户文件是否存在
    if not os.path.exists(user_path):
        social_log.error(f"混合用户文件不存在: {user_path}")
        raise FileNotFoundError(f"混合用户文件不存在: {user_path}")
    
    print(f"hydrid_user_profiles_path: {params['hybrid_user_profiles_path']}")
    
    # 检查混合用户文件是否为空
    try:
        with open(user_path, "r", encoding="utf-8") as f:
            hybrid_users = json.load(f)
            if not hybrid_users:
                social_log.error(f"混合用户文件为空: {user_path}")
                raise ValueError(f"混合用户文件为空: {user_path}")
            social_log.info(f"成功加载混合用户文件，包含{len(hybrid_users)}个用户")
    except json.JSONDecodeError:
        social_log.error(f"混合用户文件格式错误: {user_path}")
        raise ValueError(f"混合用户文件格式错误: {user_path}")
    
    return user_path


def load_post_data(params):
    """
    加载帖子数据
    
    参数:
        params: 参数字典
        
    返回:
        list: 帖子数据列表
    """
    data_format = params['data_format']
    post_path = params['post_path']
    total_news_articles = params['total_news_articles']
    round_post_num = params['round_post_num']
    num_timesteps = params['num_timesteps']
    
    if data_format == "reddit":
        social_log.info(f"Using Reddit data format from: {post_path}")
        with open(post_path, "r") as f:
            pairs = json.load(f)
    elif data_format == "twitter":
        social_log.info(f"Using Twitter data format from: {post_path}")
        with open(post_path, "r") as f:
            twitter_data = json.load(f)
        
        # 将Twitter数据转换为与Reddit格式相似的结构
        pairs = []
        for item in twitter_data:
            trigger_news = item.get("trigger_news", "")
            tweet_page = item.get("tweet_page", "")
            pair_item = {
                "RS": {
                    "title": "None",
                    "selftext": trigger_news
                },
                "RC_1": {
                    "body": tweet_page,
                    "group": "control"
                }
            }
            pairs.append(pair_item)
        social_log.info(f"Converted {len(pairs)} Twitter items to Reddit format")
    elif data_format == "twitter_raw":
        social_log.info(f"Using Twitter Raw data format from: {post_path}")
        with open(post_path, "r") as f:
            twitter_raw_data = json.load(f)
        
        pairs = []
        for item in twitter_raw_data:
            post_id = item.get("post_id", "")
            content = item.get("content", "")
            user_id = item.get("user_id", "")
            user_name = item.get("user_name", "")
            follower = item.get("follower", 0)
            location = item.get("location", "")
            timestamp = item.get("timestamp", "")
            likes = item.get("likes", 0)
            retweets = item.get("retweets", 0)
            views = item.get("views", 0)
            quotes = item.get("quotes", 0)
            
            formatted_post = f"""Tweet ID: {post_id}
            Author: {user_name} (@{user_id})
            Number of followers: {follower}
            Location: {location}
            Time: {timestamp}
            {content}
            ❤️ {likes}  🔄 {retweets}  👁️ {views}  💬 {quotes}
            """
            
            pair_item = {
                "Original Post": {
                    "title": f"Tweet from {user_name}",
                    "text": formatted_post
                }
            }
            pairs.append(pair_item)
        social_log.info(f"Converted {len(pairs)} Twitter Raw items to Reddit format")
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
    
    # 如果指定了total_news_articles参数，则限制使用的帖子数量
    if total_news_articles is not None and total_news_articles > 0:
        pairs = pairs[:total_news_articles]
        # 确保round_post_num不超过可用帖子数量
        if round_post_num > len(pairs) // num_timesteps:
            round_post_num = max(1, len(pairs) // num_timesteps)
            social_log.info(f"调整round_post_num为{round_post_num}，以适应总帖子数量{len(pairs)}")
    
    return pairs


async def create_agent_graph(params, user_path, csv_path, twitter_channel, inference_channel, 
                           cognition_space_dict, action_space_prompt, 
                           is_openai_model, is_deepseek_model, deepseek_api_base, 
                           is_local_model, local_model_api_base, 
                           multi_api_handler, inference_configs):
    """
    创建智能体图
    
    参数:
        params: 参数字典
        ... 其他参数
        
    返回:
        AgentGraph: 智能体图对象
    """
    # 检查是否使用可控用户
    if not params['controllable_user']:
        raise ValueError("Uncontrollable user is not supported")
    
    # 生成控制代理
    agent_graph, id_mapping = await gen_control_agents_with_data(
        params['data_name'],
        twitter_channel,
        control_user_num=2,
        cognition_space_dict=cognition_space_dict,
        action_space_prompt=action_space_prompt,
    )
    
    agent_graph = await generate_reddit_agents(
        params['data_name'],
        user_path,
        csv_path,
        twitter_channel,
        inference_channel,
        agent_graph,
        id_mapping,
        params['follow_post_agent'],
        params['mute_post_agent'],
        action_space_prompt,
        inference_configs["model_type"],
        is_openai_model,
        is_deepseek_model,
        deepseek_api_base,
        params['num_agents'],
        is_local_model,
        local_model_api_base,
        cognition_space_dict,
        multi_api_handler=multi_api_handler,
        max_concurrent_per_api=params['max_concurrent_per_api'],
        validate_cognitive_state=params['validate_cognitive_state'],
        max_retries=params['max_retries'],
        causal_method=params['causal_method'],
        causal_analysis_frequency=params['causal_analysis_frequency'],
        use_camel=params['use_camel'],
        max_tokens=params['max_tokens'],
        temperature=params['temperature'],
    )
    
    return agent_graph


def initialize_simulation_data(params, inference_configs, agent_graph):
    """
    初始化模拟数据
    
    参数:
        params: 参数字典
        inference_configs: 推理配置
        agent_graph: 智能体图
        
    返回:
        dict: 模拟数据字典
    """
    simulation_data = {
        "metadata": {
            "simulation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timesteps": params['num_timesteps'],
            "model_type": inference_configs["model_type"]
        },
        "agents": [],
        "posts": [],
        "comments": [],
        "actions": []
    }
    
    # 收集代理信息
    for agent_id in range(agent_graph.get_num_nodes()):
        try:
            agent = agent_graph.get_agent(agent_id)
            agent_data = {
                "agent_id": agent_id,
                "username": agent.user_info.name,
                "bio": agent.user_info.description
            }
            simulation_data["agents"].append(agent_data)
        except Exception as e:
            social_log.error(f"Error collecting agent {agent_id} data: {e}")
    
    return simulation_data


def initialize_simulation_state(agent_graph, csv_path, params):
    """
    初始化模拟运行状态
    
    参数:
        agent_graph: 智能体图
        csv_path: CSV文件路径
        params: 参数字典
        
    返回:
        tuple: 初始化的状态变量
    """
    # 维护活跃用户池和所有非控制用户的记录
    active_users_pool = set()
    all_non_controllable_agents = []
    
    # 跟踪模拟状态
    simulation_success = True
    
    start_time_0 = datetime.now()
    agents = list(agent_graph.get_agents())
    
    # 过滤出非可控用户
    for agent_id, agent in agents:
        if not agent.user_info.is_controllable:
            all_non_controllable_agents.append((agent_id, agent))
        agent.csv_path = csv_path
    
    # 计算每轮应该激活的固定用户数
    fixed_num_users_to_activate = int(len(all_non_controllable_agents) * params['activate_prob'])
    social_log.info(f"每轮将固定激活{fixed_num_users_to_activate}个用户（共{len(all_non_controllable_agents)}个非控制用户）")
    
    progress_bar = AsyncTqdmWrapper(
        total=params['num_timesteps'], 
        desc=f"ASCE模拟进度,参与智能体数:{len(all_non_controllable_agents)},激活率:{params['activate_prob']}[0/{params['num_timesteps']}轮]:\n", 
        colour="green"
    )
    
    # 创建一个全局字典用于存储所有用户的认知画像
    users_cognitive_profile_dict = {}
    
    # 用于跟踪响应进度
    completed_responses = 0
    response_count_by_round = {}
    
    # 安全检查
    if len(all_non_controllable_agents) == 0:
        social_log.error("没有可用的非控制智能体，无法进行模拟")
        raise ValueError("没有可用的非控制智能体")
    
    return (
        active_users_pool, all_non_controllable_agents, fixed_num_users_to_activate,
        simulation_success, progress_bar, users_cognitive_profile_dict,
        completed_responses, response_count_by_round, start_time_0
    )


def extract_simulation_parameters(data_params, simulation_params, inference_configs):
    """
    提取和整理模拟参数
    
    参数:
        data_params: 数据参数
        simulation_params: 模拟参数
        inference_configs: 推理配置
        
    返回:
        dict: 整理后的参数字典
    """
    params = {
        # 数据相关参数
        'data_name': data_params["data_name"],
        'db_path': data_params["db_path"],
        'real_user_path': data_params["real_user_path"],
        'random_user_path': data_params["random_user_path"],
        'hybrid_user_profiles_path': data_params["hybrid_user_profiles_path"],
        'post_path': data_params["post_path"],
        'action_space_file_path': data_params["normal_space_file_path"],
        'csv_path': data_params["csv_path"],
        'cognitive_space_path': data_params["cognitive_space_path"],
        
        # 模拟相关参数
        'recsys_type': simulation_params["recsys_type"],
        'controllable_user': simulation_params["controllable_user"],
        'allow_self_rating': simulation_params["allow_self_rating"],
        'show_score': simulation_params["show_score"],
        'max_rec_post_len': simulation_params["max_rec_post_len"],
        'refresh_rec_post_count': simulation_params["refresh_rec_post_count"],
        'activate_prob': simulation_params["activate_prob"],
        'data_format': simulation_params["data_format"],
        'max_concurrent_per_api': simulation_params["max_concurrent_per_api"],
        'validate_cognitive_state': simulation_params["validate_cognitive_state"],
        'max_retries': simulation_params["max_retries"],
        'use_camel': simulation_params.get("use_camel", False),
        'total_news_articles': simulation_params["total_news_articles"],
        'round_post_num': simulation_params["round_post_num"],
        'num_timesteps': simulation_params["num_timesteps"],
        'num_agents': simulation_params["num_agents"],
        'follow_post_agent': simulation_params["follow_post_agent"],
        'mute_post_agent': simulation_params["mute_post_agent"],
        'real_user_ratio': simulation_params["real_user_ratio"],
        'clock_factor': simulation_params["clock_factor"],
        'init_post_score': simulation_params["init_post_score"],
        'max_visible_comments': simulation_params.get("max_visible_comments", 5),
        'max_total_comments': simulation_params.get("max_total_comments", 10),
        'save_mode': simulation_params.get("save_mode", "db"),
        'num_historical_memory': simulation_params.get("num_historical_memory", 2),
        'prompt_mode': simulation_params["prompt_mode"],
        'think_mode': simulation_params["think_mode"],
        'causal_method': simulation_params.get("causal_method", "dbn_custom"),
        'causal_analysis_frequency': simulation_params.get("causal_analysis_frequency", 2),
        
        # API相关参数
        'model_type': inference_configs["model_type"],
        'is_openai_model': inference_configs["is_openai_model"],
        'is_deepseek_model': inference_configs["is_deepseek_model"],
        'is_local_model': inference_configs["is_local_model"],
        'local_model_api_base': inference_configs["local_model_api_base"],
        'max_tokens': inference_configs["max_tokens"],
        'temperature': inference_configs["temperature"],
    }
    
    return params


def setup_simulation_environment(params, random_seed):
    """
    设置模拟环境
    
    参数:
        params: 参数字典
        random_seed: 随机种子
        
    返回:
        tuple: (infra, twitter_channel, cognition_space_dict, multi_api_handler, db_path, csv_path, think_csv_path)
    """
    # 设置随机种子
    if random_seed is not None:
        social_log.info(f"Setting random seed in simulation environment: {random_seed}")
        random.seed(random_seed)
        try:
            import numpy as np
            np.random.seed(random_seed)
            social_log.info("NumPy random seed also set")
        except ImportError:
            social_log.info("NumPy not available, only Python random seed set")
    
    social_log.info(f"Using data format: {params['data_format']}")
    social_log.info(f"Using post data path: {params['post_path']}")
    
    # 加载认知空间
    cognition_space_dict = load_cognition_space(params['data_name'], params['cognitive_space_path'])
    print(json.dumps(cognition_space_dict, indent=2))
    
    # 如果数据库文件已存在，则删除它
    if os.path.exists(params['db_path']):
        os.remove(params['db_path'])
    
    # 设置模拟开始时间和时钟
    start_time = datetime(2025, 4, 9, 10, 0)
    clock = Clock(k=params['clock_factor'])
    twitter_channel = Channel()
    
    # 读取动作空间提示文件
    with open(params['action_space_file_path'], "r", encoding="utf-8") as file:
        action_space_prompt = file.read()
    
    # 为数据库文件名添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 为自定义数据库路径添加时间戳
    db_dir = os.path.dirname(params['db_path'])
    db_filename = os.path.basename(params['db_path']).split('.')[0]
    db_ext = os.path.basename(params['db_path']).split('.')[-1]
    db_path = os.path.join(db_dir, f"{db_filename}_{timestamp}.{db_ext}")
    
    csv_dir = os.path.dirname(params['csv_path'])
    csv_filename = os.path.basename(params['csv_path']).split('.')[0]
    csv_path = os.path.join(csv_dir, f"user_action_{csv_filename}_{timestamp}.csv")
    think_csv_path = os.path.join(csv_dir, f"user_action_think_{csv_filename}_{timestamp}.csv")
    
    # 创建数据库目录
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        social_log.info(f"Created database directory: {db_dir}")
    
    # 创建CSV输出目录
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
        social_log.info(f"Created CSV directory: {csv_dir}")
    
    # 创建平台对象
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        allow_self_rating=params['allow_self_rating'],
        show_score=params['show_score'],
        recsys_type=params['recsys_type'],
        max_rec_post_len=params['max_rec_post_len'],
        refresh_rec_post_count=params['refresh_rec_post_count'],
        activate_prob=params['activate_prob'],
        max_visible_comments=params['max_visible_comments'],
        max_total_comments=params['max_total_comments'],
    )
    
    # 在平台启动之前初始化环境变量，避免SANDBOX_TIME未定义的错误
    os.environ["TIME_STAMP"] = "0"
    os.environ["SANDBOX_TIME"] = "0"
    social_log.info("已初始化环境变量 TIME_STAMP 和 SANDBOX_TIME 为 0")
    
    # 设置多 API 处理器
    multi_api_handler = None
    if params['is_local_model'] and params['local_model_api_base'] and "," in params['local_model_api_base']:
        # 分割API URL
        api_urls = [url.strip() for url in params['local_model_api_base'].split(",")]
        social_log.info(f"检测到多API配置，API数量: {len(api_urls)}")
        social_log.info(f"API列表: {api_urls}")
        
        # 导入MultiApiHandler
        from asce.social_agent.api_handler import MultiApiHandler
        
        # 创建MultiApiHandler实例
        multi_api_handler = MultiApiHandler(
            api_urls=api_urls,
            max_concurrent_per_api=params['max_concurrent_per_api'],
            max_retries=params['max_retries'],
            validate_cognitive_state=params['validate_cognitive_state']
        )
        social_log.info(f"已创建MultiApiHandler，最大并发数: {params['max_concurrent_per_api']}，最大重试数: {params['max_retries']}")
        social_log.info(f"认知状态验证: {'已启用' if params['validate_cognitive_state'] else '已禁用'}")
    else:
        social_log.info(f"使用单一API配置: {params['local_model_api_base']}")
    
    return infra, twitter_channel, cognition_space_dict, multi_api_handler, db_path, csv_path, think_csv_path, action_space_prompt


# 主要运行函数，使用async关键字定义为异步函数
async def normal_running(data_params=None,
                    model_configs=None,
                    inference_configs=None,
                    simulation_params = None,
                    guidance_engine_config=None,
                    guidance_tasks_config=None,
                    random_seed=None):

    # 提取和整理参数
    params = extract_simulation_parameters(data_params, simulation_params, inference_configs)
    
    # 设置模拟环境
    infra, twitter_channel, cognition_space_dict, multi_api_handler, db_path, csv_path, think_csv_path, action_space_prompt = setup_simulation_environment(params, random_seed)




    # 创建并启动平台运行任务
    twitter_task = asyncio.create_task(infra.running())
    inference_channel = Channel()  # 创建推理通道对象
    
    # 初始化认知引导引擎 (CGE)
    cge_controller = setup_cge_engine(guidance_engine_config, guidance_tasks_config, inference_configs, infra, db_path)
    
    # 设置模型配置（兼容旧代码）
    is_openai_model = params['is_openai_model']
    is_deepseek_model = params['is_deepseek_model'] 
    is_local_model = params['is_local_model']
    deepseek_api_base = inference_configs.get("deepseek_api_base")
    local_model_api_base = params['local_model_api_base']
    
    # 如果未配置模型相关参数
    if not (is_openai_model or is_deepseek_model or is_local_model):
        social_log.warning("未配置任何模型，将使用本地模型")
        is_local_model = True

    # 初始化用户数据
    user_path = initialize_user_data(params)
    causal_json_file_path = None  # 默认值


    # 生成智能体
    agent_graph = await create_agent_graph(params, user_path, csv_path, twitter_channel, inference_channel, 
                                         cognition_space_dict, action_space_prompt, 
                                         is_openai_model, is_deepseek_model, deepseek_api_base, 
                                         is_local_model, local_model_api_base, 
                                         multi_api_handler, inference_configs)

    # 加载帖子数据
    pairs = load_post_data(params)

    # 初始化模拟数据和状态
    simulation_data = initialize_simulation_data(params, inference_configs, agent_graph)

    # 初始化模拟运行状态
    (
        active_users_pool, all_non_controllable_agents, fixed_num_users_to_activate, 
        simulation_success, progress_bar, users_cognitive_profile_dict, 
        completed_responses, response_count_by_round, start_time_0
    ) = initialize_simulation_state(agent_graph, csv_path, params)


    # 开始时间步循环 - 模拟主循环
    for timestep in range(0, params['num_timesteps'] + 1):

        # 驱动认知引导引擎
        if cge_controller:
            try:
                cge_controller.advance_timestep(timestep)
                social_log.debug(f"时间步 {timestep}: CGE引导引擎处理完成")
            except Exception as e:
                social_log.error(f"时间步 {timestep}: CGE引导引擎处理失败: {e}")

        if timestep == 0:
            # ---------- 初始化阶段 ----------
            social_log.info("======= 初始化阶段(第0轮) =======")
            init_tasks = []
            # 为所有非可控用户构建初始化任务
            for agent_id, agent in all_non_controllable_agents:
                agent.step_counter = timestep
                agent.think_mode = params['think_mode']
                agent.num_historical_memory = params['num_historical_memory']
                init_tasks.append(agent.initialize_cognitive_profile())
                init_tasks.append(agent.init_save_user_information())

            # 并发执行初始化
            if init_tasks:
                social_log.info(f"开始为{len(init_tasks)}个用户执行认知档案初始化")
                await asyncio.gather(*init_tasks)
                social_log.info("完成认知档案初始化")

                # 保存所有用户的认知画像到全局字典
                for agent_id, agent in all_non_controllable_agents:
                    # 确保cognitive_profile不为None，如果为None则初始化为默认认知档案
                    if not hasattr(agent, 'cognitive_profile') or agent.cognitive_profile is None:
                        social_log.warning(f"用户{agent_id}的认知档案为空，使用默认认知档案")
                        # 设置默认认知档案
                        await agent.initialize_cognitive_profile()

                    users_cognitive_profile_dict[agent_id] = agent.cognitive_profile.copy()
                social_log.info(f"已将{len(users_cognitive_profile_dict)}个用户的认知画像保存到全局字典")
                social_log.info("==== 保存所有智能体的初始状态（轮次为0）====")
                await save_user_actions_to_csv(all_non_controllable_agents, csv_path, think_csv_path, timestep, include_initial_state=True)
                social_log.info("初始状态保存完成")
        else:
            try:
                os.environ["TIME_STAMP"] = str(timestep)
                os.environ["SANDBOX_TIME"] = str(timestep)  # 添加SANDBOX_TIME环境变量

                # 更新进度条
                await progress_bar.update(1)
                progress_bar.set_description(f"模拟进度 ({timestep}/{params['num_timesteps']}轮)")

                social_log.info(f"======= 时间步 {timestep}/{params['num_timesteps']} =======")

                # 更新所有智能体的轮次计数器和认知画像字典引用
                for agent_id, agent in agent_graph.get_agents():
                    agent.users_cognitive_profile_dict = users_cognitive_profile_dict
                    agent.step_counter = timestep
                # 获取发帖代理/评分代理
                post_agent = agent_graph.get_agent(0)
                rate_agent = agent_graph.get_agent(1)

                await export_data(post_agent, rate_agent, pairs, timestep, params['round_post_num'], params['data_format'], params['init_post_score'])
                await infra.update_rec_table()
                social_log.info("更新推荐表完成")
                # 清空上一轮的活跃用户池
                active_users_pool.clear()
                # 再次校验非控制用户列表
                valid_agents = []
                for aid, agent in all_non_controllable_agents:
                    if agent and agent.user_info:
                        valid_agents.append((aid, agent))
                all_non_controllable_agents = valid_agents

                if not all_non_controllable_agents:
                    social_log.error(f"时间步{timestep}：所有非控制用户都无效！")
                    simulation_success = False
                    break

                # 随机挑选要激活的代理
                selected_agents = random.sample(all_non_controllable_agents, min(fixed_num_users_to_activate, len(all_non_controllable_agents)))
                for aid, _ in selected_agents:
                    active_users_pool.add(aid)

                    # 记录激活列表
                with open("active_users_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"时间步 {timestep}:\n")
                    f.write(f"总激活用户数: {len(active_users_pool)}\n")
                    f.write(f"激活用户ID: {sorted(list(active_users_pool))}\n")
                    f.write("-" * 50 + "\n")

                social_log.info(f"时间步{timestep}激活了{len(active_users_pool)}个用户: {sorted(list(active_users_pool))}")

                # 让激活的用户执行动作
                tasks = []
                causal_json_file_path = params['data_name'] + "_causal.json"  # 使用参数中的数据名称
                for aid, agent in selected_agents:
                    agent.causal_json_file_path = params['data_name'] + "_causal.json"
                    tasks.append(agent.perform_action_by_llm(save_mode=params['save_mode']))
                random.shuffle(tasks)

                if tasks:
                    social_log.info(f"执行{len(tasks)}个用户操作任务")
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # 检查执行结果
                    success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
                    error_count = sum(1 for r in results if isinstance(r, Exception))
                    social_log.info(f"用户操作执行结果: 成功{success_count}个，失败{error_count}个")
                else:
                    social_log.warning("没有用户被激活，跳过执行用户操作")

                active_agents = selected_agents
                active_ids = {aid for aid, _ in selected_agents}
                non_active_agents = [(aid, agent) for aid, agent in all_non_controllable_agents if aid not in active_ids]

                # 保存用户行为数据
                if params['save_mode'] == "db":
                    # 纯数据库保存模式：所有数据都保存到数据库
                    db_save_tasks = []
                    for aid, agent in active_agents:
                        agent.is_active = True
                        db_save_tasks.append(agent.save_user_action_dict(save_mode="db"))
                    for aid, agent in non_active_agents:
                        agent.is_active = False
                        db_save_tasks.append(agent.save_user_action_dict(save_mode="db"))

                    # 等待数据库保存完成
                    if db_save_tasks:
                        await asyncio.gather(*db_save_tasks)
                        social_log.info(f"已成功将{len(db_save_tasks)}个用户的行为数据保存到数据库")

                elif params['save_mode'] == "csv":
                    # 纯CSV保存模式：只将认知状态保存到CSV文件
                    csv_save_tasks = []
                    # 先更新每个智能体的user_action_dict
                    for aid, agent in active_agents:
                        agent.is_active = True
                        csv_save_tasks.append(agent.save_user_action_dict(save_mode="csv"))
                    for aid, agent in non_active_agents:
                        agent.is_active = False
                        csv_save_tasks.append(agent.save_user_action_dict(save_mode="csv"))

                    # 等待所有智能体更新完成user_action_dict
                    if csv_save_tasks:
                        await asyncio.gather(*csv_save_tasks)

                    # 保存到CSV文件
                    await save_user_actions_to_csv(all_non_controllable_agents, csv_path, think_csv_path, timestep, include_initial_state=False)
                    social_log.info(f"已成功将用户的认知状态保存到CSV文件")

                elif params['save_mode'] == "both":
                    # 混合保存模式：
                    # 1. 行为数据（如关注、点赞等）保存到数据库
                    # 2. 认知状态只保存在CSV文件中

                    # 先处理数据库保存（不包含认知状态）
                    db_save_tasks = []
                    for aid, agent in active_agents:
                        agent.is_active = True
                        db_save_tasks.append(agent.save_user_action_dict(save_mode="both"))
                    for aid, agent in non_active_agents:
                        agent.is_active = False
                        db_save_tasks.append(agent.save_user_action_dict(save_mode="both"))

                    # 等待数据库保存完成
                    if db_save_tasks:
                        await asyncio.gather(*db_save_tasks)
                        social_log.info(f"已成功将{len(db_save_tasks)}个用户的行为数据（不含认知状态）保存到数据库")

                    # 再处理CSV保存（包含认知状态）
                    csv_save_tasks = []
                    # 先更新每个智能体的user_action_dict
                    for aid, agent in active_agents:
                        agent.is_active = True
                        csv_save_tasks.append(agent.save_user_action_dict(save_mode="csv"))
                    for aid, agent in non_active_agents:
                        agent.is_active = False
                        csv_save_tasks.append(agent.save_user_action_dict(save_mode="csv"))

                    # 等待所有智能体更新完成user_action_dict
                    if csv_save_tasks:
                        await asyncio.gather(*csv_save_tasks)

                    # 保存到CSV文件
                    await save_user_actions_to_csv(all_non_controllable_agents, csv_path, think_csv_path, timestep, include_initial_state=False)
                    social_log.info(f"已成功将用户的认知状态保存到CSV文件")

                else:
                    social_log.warning(f"未知的保存模式: {params['save_mode']}，跳过保存用户行为数据")


                # # 保存认知状态和因果分析数据
                # memory_tasks = []
                # for aid, agent in active_agents:
                #     memory_tasks.append(agent.save_cognitive_state_to_json(causal_json_file_path))

                # for aid, agent in non_active_agents:
                #     #memory_tasks.append(agent.save_memory_data(timestep, False))
                #     memory_tasks.append(agent.save_cognitive_state_to_json(causal_json_file_path))

                # 更新所有智能体的轮次计数器和认知画像字典引用

                # if memory_tasks:
                #     try:
                #         await asyncio.gather(*memory_tasks)
                #         social_log.info(f"已保存{len(memory_tasks)}个用户的记忆数据")
                #     except Exception as e:
                #         social_log.error(f"保存记忆数据时出错: {e}")

                for agent_id, agent in agent_graph.get_agents():
                    users_cognitive_profile_dict = agent.users_cognitive_profile_dict

                # 如果是第1轮，重新计算时钟因子
                if timestep == 1:
                    time_difference = datetime.now() - start_time_0
                    two_hours_in_seconds = timedelta(hours=2).total_seconds()
                    clock_factor = two_hours_in_seconds / time_difference.total_seconds()
                    # 注意: 在模块化版本中，clock对象需要从环境中获取
                    # 这里的clock对象在setup_simulation_environment中创建
                    # 但由于原始代码的限制，这里保留原有逻辑，可能需要进一步优化
                    social_log.info(f"clock_factor重设为: {clock_factor}")
                    social_log.warning("注意: clock对象的更新在模块化版本中可能需要调整")

                # 测试数据库中本轮行为数据
                db_conn = sqlite3.connect(db_path)
                cursor = db_conn.cursor()
                # 这里可以执行查询或别的检查
                db_conn.close()

                # 确保所有缓存数据都被保存到数据库
                await infra._flush_all_caches()

                # 记录本轮响应数
                response_count_by_round[timestep] = len(active_users_pool)
                social_log.info(f"时间步{timestep}完成了{len(active_users_pool)}个响应，累计完成{completed_responses}/{len(active_users_pool)}")

            except Exception as e:
                social_log.error(f"模拟运行出错(第{timestep}轮): {e}")
                simulation_success = False
                break

    # 模拟结束时，确保所有缓存数据都被保存到数据库
    try:
        await infra._flush_all_caches()
        social_log.info("模拟结束时成功将所有缓存数据保存到数据库")
    except Exception as e:
        social_log.error(f"模拟结束时保存缓存数据出错: {e}")

    social_log.info(f"模拟数据已保存为数据库: {db_path}")
    social_log.info("==== 模拟完成状态报告 ====")
    social_log.info(f"执行时间步: {params['num_timesteps']}")
    social_log.info(f"模拟状态: {'成功' if simulation_success else '有错误'}")

    try:
        # 连接数据库获取最终状态
        db_conn = sqlite3.connect(db_path)
        cursor = db_conn.cursor()

        # 检查think表中最大的时间步和每个时间步的记录数
        cursor.execute("SELECT MAX(step_number) FROM think")
        max_step = cursor.fetchone()[0]
        social_log.info(f"数据库中最大时间步: {max_step}")

        cursor.execute("SELECT step_number, COUNT(*) FROM think GROUP BY step_number ORDER BY step_number")
        step_counts = cursor.fetchall()
        social_log.info(f"各时间步记录数: {step_counts}")

        # 最后关闭连接
        db_conn.close()
    except Exception as e:
        social_log.error(f"生成状态报告时出错: {e}")

    # 终止平台任务
    twitter_task.cancel()

    try:
        await twitter_task
    except asyncio.CancelledError:
        pass

    # 关闭进度条
    await progress_bar.close()

    print(f"\n{Fore.GREEN}===== 模拟完成 ====={Fore.RESET}")
    print(f"数据库文件: {db_path}")
    print(f"CSV文件: {csv_path}")
    print(f"CSV THINK文件: {think_csv_path}")


    # 打印响应质量统计信息
    try:
        from asce.social_agent.utils.response_quality_utils import print_response_quality_stats
        print_response_quality_stats(save_to_file=True)
        social_log.info("已打印响应质量统计信息")
    except Exception as e:
        social_log.error(f"打印响应质量统计信息时出错: {e}")

    # 关闭认知引导引擎
    if cge_controller:
        try:
            cge_controller.shutdown()
            social_log.info("认知引导引擎已关闭")
        except Exception as e:
            social_log.error(f"关闭认知引导引擎失败: {e}")

    social_log.info("Simulation finish!")


# 主程序入口
if __name__ == "__main__":
    # 禁用所有与异步相关的运行时警告
    warnings.simplefilter("ignore", RuntimeWarning)

    # 设置异步日志为ERROR级别，抑制WARNING信息
    asyncio.get_event_loop().set_debug(False)
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    args = parser.parse_args()  # 解析命令行参数
    
    # 初始化日志系统
    log_dir, platform_log, agent_log, social_log = setup_logging()
    
    # 加载配置文件并保存到日志目录
    cfg = load_and_save_config(args.config_path)
    
    # 设置随机种子
    random_seed = setup_random_seed(args, args.config_path)


    # 获取各部分配置
    data_params = cfg.get("data", {})  # 获取数据参数
    simulation_params = cfg.get("simulation", {})  # 获取模拟参数
    model_configs = cfg.get("model", {})  # 获取模型配置
    inference_params = cfg.get("inference", {})  # 获取推理参数
    guidance_engine_config = cfg.get("guidance_engine", {})  # 获取引导引擎配置
    guidance_tasks_config = cfg.get("guidance_tasks", [])  # 获取引导任务配置
    
    # 打印配置信息（简化版）
    print(f"{Fore.CYAN}===== ASCE系统配置信息 ====={Fore.RESET}")
    print(f"数据集: {data_params.get('data_name', 'N/A')}")
    print(f"模型类型: {inference_params.get('model_type', 'N/A')}")
    print(f"智能体数量: {simulation_params.get('num_agents', 'N/A')}")
    print(f"时间步数: {simulation_params.get('num_timesteps', 'N/A')}")
    print(f"日志目录: {log_dir}")
    print(f"{Fore.CYAN}==============================={Fore.RESET}")
    # 检查是否使用个体模拟模式
    individual_mode = simulation_params.get("individual_mode", False)
    label_mode = simulation_params.get("label_mode", False)
    print(f"individual_mode: {individual_mode}, label_mode: {label_mode}")
    print(f"{Fore.GREEN}Using Standard Mode: {Fore.RESET}")
    
    # 运行标准模拟
    asyncio.run(
        normal_running(
            data_params=data_params,
            model_configs=model_configs,
            inference_configs=inference_params,
            simulation_params=simulation_params,
            guidance_engine_config=guidance_engine_config,  # 传递引导引擎配置
            guidance_tasks_config=guidance_tasks_config,    # 传递引导任务配置
            random_seed=random_seed,  # 传递随机种子
        ),
        debug=True,  # 启用调试模式
    )
    print(f"Simulation Finish !!")
    # 输出日志路径、保存的文件路径


