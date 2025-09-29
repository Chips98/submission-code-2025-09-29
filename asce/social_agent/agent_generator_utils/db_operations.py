"""数据库操作模块,用于处理代理生成过程中的数据库操作。

包含用户注册、关注关系处理、帖子管理等功能。
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from asce.social_agent.agent import SocialAgent
from asce.social_agent.agent_environment import Environment

logger = logging.getLogger(__name__)

async def register_user(
    env: Environment,
    user_info: Dict[str, Any],
    model_type: str,
    model_config: Dict[str, Any],
    temperature: float,
    agent_id: str,
    logger: Optional[logging.Logger] = None
) -> SocialAgent:
    """注册单个用户,返回代理实例"""
    agent = SocialAgent(
        agent_id=agent_id,
        user_info=user_info,
        model_type=model_type,
        model_config=model_config,
        temperature=temperature,
        env=env,
        logger=logger
    )
    await env.register_user(agent)
    return agent

async def register_users_batch(
    env: Environment,
    user_infos: List[Dict[str, Any]],
    model_types: List[str],
    model_configs: List[Dict[str, Any]],
    temperatures: List[float],
    agent_ids: List[str],
    logger: Optional[logging.Logger] = None
) -> List[SocialAgent]:
    """批量注册用户,返回代理实例列表"""
    tasks = []
    for user_info, model_type, model_config, temperature, agent_id in zip(
        user_infos, model_types, model_configs, temperatures, agent_ids
    ):
        tasks.append(
            register_user(
                env, user_info, model_type, model_config, 
                temperature, agent_id, logger
            )
        )
    return await asyncio.gather(*tasks)

async def setup_follow_relations(
    env: Environment,
    agents: List[SocialAgent],
    follow_matrix: List[List[int]]
) -> None:
    """设置代理之间的关注关系"""
    for i, agent in enumerate(agents):
        for j, follow in enumerate(follow_matrix[i]):
            if follow == 1:
                await env.follow_user(agent.agent_id, agents[j].agent_id)

async def create_initial_posts(
    env: Environment,
    agents: List[SocialAgent],
    post_contents: List[str]
) -> None:
    """创建初始帖子"""
    tasks = []
    for agent, content in zip(agents, post_contents):
        tasks.append(env.create_post(agent.agent_id, content))
    await asyncio.gather(*tasks)

async def execute_batch_operations(
    env: Environment,
    operations: List[Tuple[str, Dict[str, Any]]]
) -> List[Any]:
    """执行批量操作,返回操作结果列表"""
    tasks = []
    for op_type, op_args in operations:
        if op_type == "register":
            tasks.append(env.register_user(**op_args))
        elif op_type == "follow":
            tasks.append(env.follow_user(**op_args))
        elif op_type == "post":
            tasks.append(env.create_post(**op_args))
        else:
            logger.warning(f"未知的操作类型: {op_type}")
    return await asyncio.gather(*tasks) 