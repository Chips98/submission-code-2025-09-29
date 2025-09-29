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
from __future__ import annotations

import ast
import asyncio
import json
import logging
import random
from typing import Any, List

import numpy as np
import pandas as pd
import tqdm
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole

from asce.social_agent import AgentGraph, SocialAgent
from asce.social_platform import Channel, Platform
from asce.social_platform.config import Neo4jConfig, UserInfo
from asce.social_agent.visualization import visualize_agent_graph

# 获取社交模块的日志记录器
social_log = logging.getLogger(name="social")

async def gen_control_agents_with_data(
    data_name: None,
    channel: Channel,
    control_user_num: int,
    cognition_space_dict: dict = None,
    action_space_prompt: str = None,  # 添加动作空间提示参数
) -> tuple[AgentGraph, dict]:
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        user_info = UserInfo(
            is_controllable=True,
            profile={
                "other_info": {
                    "user_profile": "None",
                    "gender": "None",
                    "mbti": "None",
                    "country": "None",
                    "age": "None",
                    # 添加缺失的字段
                    "profession": "None",
                    "interested_topics": ["None"],
                    "influence_metrics": {
                        "like_count": 0,
                        "retweet_count": 0,
                        "influence_score": 0
                    }
                }
            },
            recsys_type="reddit",
        )
        csv_path = ''
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(data_name, i, csv_path, user_info, channel, agent_graph=agent_graph, cognition_space_dict=cognition_space_dict, action_space_prompt=action_space_prompt)  # 传递动作空间提示
        # Add agent to the agent graph
        agent_graph.add_agent(agent)
        user_name = "momo"
        name = "momo"
        bio = "None."

        # 添加调试输出
        #print(f"==DEBUG== 正在注册用户: agent_id={i}, user_name={user_name}")

        try:
            response = await agent.env.action.sign_up(user_name, name, bio)

            # 添加调试输出
            #print(f"==DEBUG== 注册结果: {response}")

            # 检查注册是否成功
            if response.get("success", False):
                user_id = response.get("user_id")
                if user_id is not None:
                    agent_user_id_mapping[i] = user_id
                    #print(f"==DEBUG== 用户注册成功: agent_id={i}, user_id={user_id}")
                else:
                    print(f"==DEBUG== 警告: 用户注册成功但未返回user_id, agent_id={i}")
                    # 使用agent_id作为备用
                    agent_user_id_mapping[i] = i
            else:
                print(f"==DEBUG== 错误: 用户注册失败, agent_id={i}, 错误信息: {response.get('error', '未知错误')}")
                # 使用agent_id作为备用
                agent_user_id_mapping[i] = i
        except Exception as e:
            print(f"==DEBUG== 异常: 用户注册过程中发生错误, agent_id={i}, 错误: {str(e)}")
            import traceback
            print(f"==DEBUG== 错误详情: {traceback.format_exc()}")
            # 使用agent_id作为备用
            agent_user_id_mapping[i] = i

    return agent_graph, agent_user_id_mapping


async def generate_reddit_agents(
    data_name: str,
    agent_info_path: str,
    csv_path: str,
    twitter_channel: Channel,
    inference_channel: Channel,
    agent_graph: AgentGraph | None = AgentGraph,
    agent_user_id_mapping: dict[int, int] | None = None,
    follow_post_agent: bool = False,
    mute_post_agent: bool = False,
    action_space_prompt: str = None,
    model_type: str = "llama3.1-8b",
    is_openai_model: bool = False,
    is_deepseek_model: bool = False,
    deepseek_api_base: str = None,
    num_agents: int = None,
    is_local_model: bool = False,
    local_model_api_base: str = None,
    cognition_space_dict: dict = None,
    multi_api_handler = None,  # 添加多API处理器参数
    max_concurrent_per_api: int = 64,  # 添加并发限制参数
    validate_cognitive_state: bool = False,  # 添加验证认知状态参数
    max_retries: int = 3,  # 添加最大重试次数参数
    causal_method: str = "dbn_custom",  # 添加因果分析方法参数
    causal_analysis_frequency: int = 2,  # 添加因果分析频率参数
    use_camel: bool = False, 
    max_tokens: int = 32000, 
    temperature: float = 0.5,

) -> AgentGraph:
    """
    生成Reddit平台的代理用户群体

    这个函数用于从提供的代理信息文件中创建Reddit平台的虚拟用户，并设置它们的属性和关系。
    如果指定了num_agents参数，则只创建指定数量的代理。
    """
    # 如果没有提供代理ID到用户ID的映射，则创建一个空字典
    if agent_user_id_mapping is None:
        agent_user_id_mapping = {}
    # 如果没有提供代理关系图，则创建一个新的
    if agent_graph is None:
        agent_graph = AgentGraph()

    # 获取当前代理图中已有的节点数量（控制用户数量）
    control_user_num = agent_graph.get_num_nodes()
    social_log.info(f"当前已有{control_user_num}个控制用户")

    # 从文件中读取代理信息（JSON格式）
    try:
        with open(agent_info_path, "r") as file:
            agent_info = json.load(file)
        social_log.info(f"成功从{agent_info_path}读取了{len(agent_info)}个代理信息")
    except Exception as e:
        social_log.error(f"读取代理信息文件{agent_info_path}时出错: {e}")
        raise

    # 如果指定了num_agents参数，限制创建的代理数量
    original_agent_count = len(agent_info)
    if num_agents is not None and num_agents > 0:
        # 确保不超过可用的代理信息数量
        max_available = len(agent_info)
        if num_agents > max_available:
            social_log.warning(f"请求的代理数量({num_agents})超过了可用的代理信息数量({max_available})，将使用所有可用代理")
            num_agents = max_available
        # 只使用指定数量的代理信息
        agent_info = agent_info[:num_agents]
        social_log.info(f"将创建{len(agent_info)}个代理，从原始的{original_agent_count}个代理信息中选择")

    # 创建用户名到用户ID的映射字典
    username_to_user_id = {}

    # 预处理计数器
    successful_agents = 0
    failed_agents = 0
    invalid_agents = []

    # 定义一个异步函数，用于处理单个代理的创建和设置
    async def process_agent(i):
        nonlocal successful_agents, failed_agents

        try:
            # 实例化一个代理
            profile = {
                "nodes": [],  # 与其他代理的关系节点列表
                "edges": [],  # 关系详情列表
                "other_info": {},  # 其他信息的字典
            }
            # 使用代理信息更新个人资料
            profile["other_info"]["user_profile"] = agent_info[i]["persona"]  # 设置用户个性描述
            profile["other_info"]["mbti"] = agent_info[i]["mbti"]  # 设置MBTI人格类型
            profile["other_info"]["gender"] = agent_info[i]["gender"]  # 设置性别
            profile["other_info"]["age"] = agent_info[i]["age"]  # 设置年龄
            profile["other_info"]["country"] = agent_info[i]["country"]  # 设置国家

            # 保存关注列表和粉丝列表
            if "follow_list" in agent_info[i]:
                profile["other_info"]["follow_list"] = agent_info[i]["follow_list"]
            if "follower_list" in agent_info[i]:
                profile["other_info"]["follower_list"] = agent_info[i]["follower_list"]

            # 如果用户数据中包含认知维度信息，则添加到个人配置中
            if "cognitive_profile" in agent_info[i]:
                # 创建符合DEFAULT_COGNITIVE_PROFILE结构的认知配置文件
                cognitive_profile = {
                    'mood': {'type': 'neutral', 'value': 'neutral'},
                    'emotion': {'type': 'neutral', 'value': 'neutral'},
                    'stance': {'type': 'neutral', 'value': 'neutral'},
                    'thinking': {'type': 'neutral', 'value': 'neutral'},
                    'intention': {'type': 'neutral', 'value': 'neutral'},
                    'opinion': {
                        'viewpoint_1': 'Indifferent',
                        'viewpoint_2': 'Indifferent',
                        'viewpoint_3': 'Indifferent',
                        'viewpoint_4': 'Indifferent',
                        'viewpoint_5': 'Indifferent',
                        'viewpoint_6': 'Indifferent'
                    }
                }

                # 从用户数据中提取认知信息
                user_cognitive = agent_info[i]["cognitive_profile"]

                # 处理基本认知字段
                for field in ['mood', 'emotion', 'stance', 'thinking', 'intention']:
                    if field in user_cognitive and isinstance(user_cognitive[field], dict):
                        if 'type' in user_cognitive[field]:
                            cognitive_profile[field]['type'] = user_cognitive[field]['type']
                        if 'value' in user_cognitive[field]:
                            cognitive_profile[field]['value'] = user_cognitive[field]['value']

                # 处理观点字段 - 从列表格式转换为字典格式
                if 'opinion' in user_cognitive and isinstance(user_cognitive['opinion'], list):
                    for opinion_item in user_cognitive['opinion']:
                        if isinstance(opinion_item, dict):
                            for key in opinion_item:
                                if key.startswith('viewpoint_'):
                                    support_level = opinion_item.get('type_support_levels', 'Indifferent')
                                    cognitive_profile['opinion'][key] = support_level

                # 记录转换后的认知档案
                social_log.debug(f"用户{agent_info[i]['username']}的认知档案: {cognitive_profile}")

                # 添加认知维度信息到个人配置中
                profile["other_info"]["cognitive_profile"] = cognitive_profile

            # 创建用户信息对象
            user_info = UserInfo(
                name=agent_info[i]["username"],  # 用户名
                description=agent_info[i]["bio"],  # 用户简介
                profile=profile,  # 用户详细资料
                recsys_type="reddit",  # 推荐系统类型为reddit
            )

            # 创建社交代理实例
            agent_id = i + control_user_num  # 代理ID为索引加上控制用户数量
            agent = SocialAgent(
                data_name=data_name,
                agent_id=agent_id,  # 代理ID为索引加上控制用户数量
                csv_path = csv_path,
                user_info=user_info,  # 用户信息
                twitter_channel=twitter_channel,  # Twitter通信渠道
                inference_channel=inference_channel,  # 推理通道
                model_type=model_type,  # 模型类型
                agent_graph=agent_graph,  # 代理关系图
                action_space_prompt=action_space_prompt,  # 动作空间提示
                is_openai_model=is_openai_model,  # 是否使用OpenAI模型
                is_deepseek_model=is_deepseek_model,  # 是否使用DeepSeek模型
                deepseek_api_base=deepseek_api_base,  # DeepSeek API基础URL
                is_local_model=is_local_model,  # 是否使用本地模型
                local_model_api_base=local_model_api_base,  # 本地模型API基础URL
                cognition_space_dict=cognition_space_dict,  # 认知空间字典
                multi_api_handler=multi_api_handler,  # 多API处理器
                max_concurrent_per_api=max_concurrent_per_api,  # 每个API最大并发请求数
                validate_cognitive_state=validate_cognitive_state,  # 是否验证认知状态
                max_retries=max_retries,  # 最大重试次数
                causal_method=causal_method,  # 因果分析方法
                causal_analysis_frequency=causal_analysis_frequency,  # 因果分析频率
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # 设置是否使用Camel框架的响应和解析方式
            if use_camel:
                agent.use_camel = True
                social_log.info(f"代理{agent_id}({agent_info[i]['username']})启用了Camel框架的响应和解析方式")

            # 将代理添加到代理关系图中
            agent_graph.add_agent(agent)

            # 注册代理并将其信息添加到数据库
            social_log.debug(f"正在注册代理{agent_id}({agent_info[i]['username']})...")
            response = await agent.env.action.sign_up(agent_info[i]["username"],  # 使用用户名注册
                                                    agent_info[i]["realname"],  # 真实姓名
                                                    agent_info[i]["bio"])  # 个人简介

            # 验证注册是否成功
            if response and "user_id" in response:
                user_id = response["user_id"]  # 获取注册后的用户ID
                agent_user_id_mapping[agent_id] = user_id  # 将代理ID映射到用户ID
                # 将用户名映射到用户ID，用于后续处理关注关系
                username_to_user_id[agent_info[i]["username"]] = user_id
                successful_agents += 1
                social_log.debug(f"代理{agent_id}({agent_info[i]['username']})注册成功，用户ID:{user_id}")

                # 保存用户详细信息到user_information表
                try:
                    await agent.save_user_information()
                    social_log.debug(f"代理{agent_id}的详细信息已保存到user_information表")
                except Exception as e:
                    social_log.error(f"保存代理{agent_id}的详细信息时出错: {str(e)}")
            else:
                social_log.error(f"代理{agent_id}({agent_info[i]['username']})注册失败，响应:{response}")
                failed_agents += 1
                invalid_agents.append(agent_id)

            # 如果启用了关注发帖代理的功能
            if follow_post_agent:
                await agent.env.action.follow(1, step_number=0)  # 关注用户ID为1的用户，设置轮次为0表示初始化
                # 创建关注操作的内容记录
                content = """
{
    "reason": "He is my friend, and I would like to follow him "
              "on social media.",
    "functions": [
        {
            "name": "follow",
            "arguments": {
                "user_id": 1
            }
        }
    ]
}
"""
                # 创建助手消息并记录到代理的记忆中
                agent_msg = BaseMessage.make_assistant_message(
                    role_name="Assistant", content=content)
                agent.memory.write_record(
                    MemoryRecord(message=agent_msg, role_at_backend=OpenAIBackendRole.ASSISTANT))
            # 如果启用了屏蔽发帖代理的功能
            elif mute_post_agent:
                await agent.env.action.mute(1)  # 屏蔽用户ID为1的用户
                # 创建屏蔽操作的内容记录（注意：这里的JSON格式似乎不完整）
                content = """
{
    "reason": "He is my enemy, and I would like to mute him on social media.",
    "functions": [{
        "name": "mute",
        "arguments": {
            "user_id": 1
        }
}
"""
                # 创建助手消息并记录到代理的记忆中
                agent_msg = BaseMessage.make_assistant_message(
                    role_name="Assistant", content=content)
                agent.memory.write_record(
                    MemoryRecord(message=agent_msg, role_at_backend=OpenAIBackendRole.ASSISTANT))

        except Exception as e:
            social_log.error(f"处理代理{i}时出错: {e}")
            import traceback
            social_log.error(f"错误详情: {traceback.format_exc()}")
            failed_agents += 1
            # 如果代理ID已生成，添加到无效代理列表
            agent_id = i + control_user_num
            if agent_id not in invalid_agents:
                invalid_agents.append(agent_id)

    # 为每个代理创建处理任务
    tasks = [process_agent(i) for i in range(len(agent_info))]

    # 并发执行所有代理处理任务
    social_log.info(f"开始创建{len(tasks)}个代理...")
    await asyncio.gather(*tasks)
    social_log.info(f"代理创建完成，成功:{successful_agents}，失败:{failed_agents}")

    # 验证生成的代理数量
    actual_num_agents = agent_graph.get_num_nodes()
    expected_num_agents = control_user_num + len(agent_info)
    if actual_num_agents != expected_num_agents:
        social_log.error(f"代理数量不匹配:期望{expected_num_agents},实际{actual_num_agents}")
        # 不再抛出异常，避免中断流程，而是记录详细信息
        social_log.error(f"无效的代理ID列表: {invalid_agents}")

    # 验证代理状态，移除无效代理
    valid_agents_count = 0
    checked_agents = []
    for agent_id in range(control_user_num, actual_num_agents):
        try:
            agent = agent_graph.get_agent(agent_id)
            # 检查代理是否有效
            if agent and agent.user_info:
                valid_agents_count += 1
                checked_agents.append(agent_id)
            else:
                social_log.error(f"代理{agent_id}状态异常")
                if agent_id not in invalid_agents:
                    invalid_agents.append(agent_id)
        except Exception as e:
            social_log.error(f"验证代理{agent_id}时出错: {e}")
            if agent_id not in invalid_agents:
                invalid_agents.append(agent_id)

    social_log.info(f"代理验证完成，有效代理:{valid_agents_count}，无效代理:{len(invalid_agents)}")
    if invalid_agents:
        social_log.info(f"无效代理ID: {sorted(invalid_agents)}")
    if checked_agents:
        social_log.info(f"已检查的代理ID: {len(checked_agents)}个，范围:{min(checked_agents)}-{max(checked_agents)}")

    # 处理用户的初始关注关系
    social_log.info("开始处理用户的初始关注关系...")
    follow_tasks = []
    for i in range(len(agent_info)):
        agent_id = i + control_user_num
        # 跳过无效代理
        if agent_id in invalid_agents:
            continue

        try:
            agent = agent_graph.get_agent(agent_id)
            if agent and agent.user_info and "follow_list" in agent_info[i]:
                follow_list = agent_info[i]["follow_list"]
                username = agent_info[i]["username"]
                social_log.debug(f"【初始化关注信息】处理用户{username}(ID:{agent_id})的关注列表: {follow_list}")

                for follow_username in follow_list:
                    if follow_username in username_to_user_id:
                        follow_user_id = username_to_user_id[follow_username]
                        social_log.debug(f"【初始化关注信息】用户{username}(ID:{agent_id})将关注{follow_username}(ID:{follow_user_id})")
                        # 添加异步关注任务
                        follow_tasks.append(process_follow(agent, username, follow_username, follow_user_id))
                    else:
                        social_log.warning(f"【初始化关注信息】找不到用户{follow_username}的ID，用户{username}无法关注该用户")
        except Exception as e:
            social_log.error(f"【初始化关注信息】处理用户{agent_id}的关注关系时出错: {e}")

    # 定义处理关注的异步函数
    async def process_follow(agent, username, follow_username, follow_user_id):
        try:
            # 执行关注操作，指定轮次为0（初始化阶段）
            await agent.env.action.follow(follow_user_id, step_number=0)
            social_log.debug(f"用户{username}成功关注了{follow_username}(ID:{follow_user_id})，轮次:0(初始化)")

            # 创建关注操作的内容记录
            content = f"""
{{
    "reason": "I am interested in {follow_username}'s content and would like to follow them.",
    "functions": [
        {{
            "name": "follow",
            "arguments": {{
                "user_id": {follow_user_id}
            }}
        }}
    ]
}}
"""
            # 创建助手消息并记录到代理的记忆中
            agent_msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.memory.write_record(
                MemoryRecord(message=agent_msg, role_at_backend=OpenAIBackendRole.ASSISTANT))
            return True
        except Exception as e:
            social_log.error(f"用户{username}关注{follow_username}(ID:{follow_user_id})时出错: {e}")
            return False

    # 执行所有关注任务
    if follow_tasks:
        social_log.info(f"开始执行{len(follow_tasks)}个关注任务...")
        follow_results = await asyncio.gather(*follow_tasks)
        successful_follows = sum(1 for result in follow_results if result)
        social_log.info(f"关注任务完成，成功:{successful_follows}，失败:{len(follow_tasks) - successful_follows}")
    else:
        social_log.info("没有需要处理的关注任务")


    # 返回构建好的代理关系图
    return agent_graph

