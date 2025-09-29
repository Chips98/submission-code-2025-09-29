from __future__ import annotations
import re
import inspect
import json
import logging
import os
import pandas as pd
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import asyncio
import time
import uuid
import aiohttp
import ssl
import pdb
from camel.configs import ChatGPTConfig
from camel.memories import (ChatHistoryMemory, MemoryRecord,
                            ScoreBasedContextCreator)
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter
from asce.causal.causal_analysis_new import get_causal_relations_for_simulation

# 导入MultiApiHandler类
from asce.social_agent.api_handler import MultiApiHandler

# 加载认知规范化映射文件
NORMALIZATION_MAP_PATH = os.path.join(os.path.dirname(__file__), "cognitive_normalization.json")
try:
    with open(NORMALIZATION_MAP_PATH, 'r', encoding='utf-8') as f:
        NORMALIZATION_MAP = json.load(f)
    # 创建日志记录器在这里可能还不行，先使用print
    print(f"成功加载认知规范化映射: {NORMALIZATION_MAP_PATH}")
except Exception as e:
    print(f"加载认知规范化映射失败: {e}，将使用内置默认映射")
    NORMALIZATION_MAP = {}

from asce.social_agent.agent_action import SocialAction
from asce.social_agent.agent_environment import SocialEnvironment
from asce.social_platform import Channel
from asce.social_platform.config import UserInfo
from asce.social_agent.response_parser import ResponseParser
from asce.social_agent.parse_stats import ParseStats
from asce.social_agent.response_quality import ResponseQualityTracker
from asce.social_agent.prompt.generate_prompt import generate_prompt

if TYPE_CHECKING:
    from asce.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")
    #agent_log.propagate = False  # 添加此行，禁止日志传播到控制台
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/social.agent-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    agent_log.addHandler(file_handler)

# 全局变量用于记录因果模型计算时间
causal_model_timing = {
    'dbn_custom': [],
    'dbn_neural': [],
    'dbn_forest': []
}

# 全局变量用于记录因果模型计算的数值结果
causal_model_results = {
    'dbn_custom': [],
    'dbn_neural': [],
    'dbn_forest': []
}

#进行修改
class SocialAgent:
    r"""Social Agent."""

    # 添加默认认知档案作为类变量
    DEFAULT_COGNITIVE_PROFILE = {
        'mood': {'type': 'fail', 'value': 'fail'},
        'emotion': {'type': 'fail', 'value': 'fail'},
        'stance': {'type': 'fail', 'value': 'fail'},
        'thinking': {'type': 'fail', 'value': 'fail'},
        'intention': {'type': 'fail', 'value': 'fail'},
        'opinion': {
            'viewpoint_1': 'fail',
            'viewpoint_2': 'fail',
            'viewpoint_3': 'fail',
            'viewpoint_4': 'fail',
            'viewpoint_5': 'fail',
            'viewpoint_6': 'fail'
        }
    }


    def __init__(
        self,
        data_name: str,
        agent_id: int,
        csv_path: csv_path,
        user_info: UserInfo,
        twitter_channel: Channel,
        inference_channel: Channel = None,
        model_type: str = "llama3.1-8b",
        agent_graph: "AgentGraph" = None,
        action_space_prompt: str = None,
        is_openai_model: bool = False,
        is_deepseek_model: bool = False,
        deepseek_api_base: str = None,
        is_local_model: bool = False,
        local_model_api_base: str = None,
        cognition_space_dict: dict = None,
        multi_api_handler: Optional[MultiApiHandler] = None,  # 添加MultiApiHandler参数
        max_concurrent_per_api: int = 64,  # 添加每个API的最大并发请求数参数
        validate_cognitive_state: bool = False,  # 添加是否验证认知状态参数
        max_retries: int = 3,  # 添加最大重试次数参数
        causal_method: str = "dbn_custom",  # 添加因果分析方法参数
        causal_analysis_frequency: int = 2,  # 添加因果分析频率参数
        max_tokens: int = 32000,
        temperature: float = 0.5,
    ):
        """
        初始化社交代理对象
        """
        # 初始化代理基本属性
        self.data_name=data_name,
        self.agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.inference_channel = inference_channel
        self.model_type = model_type
        self.agent_graph = agent_graph
        self.is_openai_model = is_openai_model
        self.is_deepseek_model = is_deepseek_model
        self.deepseek_api_base = deepseek_api_base
        self.is_local_model = is_local_model
        self.local_model_api_base = local_model_api_base
        self.user_name = user_info.name
        self.cognition_space_dict = cognition_space_dict  # 保存认知空间字典

        # 保存多API处理器和相关参数
        self.multi_api_handler = multi_api_handler
        self.max_concurrent_per_api = max_concurrent_per_api
        self.validate_cognitive_state = validate_cognitive_state
        self.max_retries = max_retries
        self.causal_method = causal_method  # 保存因果分析方法
        self.causal_analysis_frequency = causal_analysis_frequency  # 保存因果分析频率
        self.individual_context = None
        self.target_context = None
        self.max_tokens = max_tokens,
        self.temperature = temperature,

        # 使用统一的日志记录器
        self.is_active = False

        # 初始化认知档案和思考记录
        self.cognitive_profile = None
        # = None
        self.user_action_dict = {}
        self.user_action_dict_think = {}
        # 初始化思考记录和行为跟踪
        self.think_record = []
        self.reason = "No Reason"  # 初始理由
        self.current_post_id = 0  # 跟踪当前查看的帖子ID
        self.current_action = {
            "action_name": "initial",
            "arguments": {},
            "reason": "initial"
        }
        self.step_counter = 0
        self.causal_json_file_path = ''
        self.raw_response = ""
        self.prompt = ''
        # 初始化行动空间
        self.action_space_prompt = action_space_prompt
        self.use_camel = False
        self.think_content = ""
        self.is_hisim = False
        self.num_historical_memory = 2
        # 初始化认知图
        self._init_parse_stats()
        # 注册通道
        if self.twitter_channel:
            self.twitter_channel.platform = self.get_platform_from_channel()

        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))
        self.history_memory = "No historical memory."  # 初始化为默认值，避免None值
        self.think_mode = ''
        self.causal_prompt = """According to existing theories, it is known that there are the following influence relationships among cognitive dimensions.
Mood is long-term and stable, mainly determined by the mood state of the previous round and external information.
Emotion is short-term and volatile, affected by current mood, one's own state in the previous round, and the environment.
Thinking decision-making is driven by Emotion and accumulates over the long term, also reflecting cognitive inertia.
Stance is directly determined by cognitive outlook.
Intention is the behavioral manifestation of Stance.
The influence of the external environment cannot be quantified. Please think about it on your own."""
        self.csv_path = csv_path
        #self.memory_manager = MemoryManager()
        self.current_posts = None
        self.agent_graph = agent_graph

        # 创建长期记忆存储目录
        self.memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory_data")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_file = os.path.join(self.memory_dir, f"agent_{self.agent_id}_memory.txt")

        # 尝试从channel获取平台对象
        self.get_platform_from_channel()

        # 异步初始化完成后，设置一个标志以便后续保存初始认知状态
        self.initial_state_saved = False

        # 设置初始化标志，用于延迟保存用户信息
        self.is_initialized = True

    def _init_parse_stats(self):
        try:
            # 初始化解析重试相关配置
            self.parse_retries = getattr(self, 'parse_retries', 2)  # 解析失败时的最大重试次数
            self.collect_parse_stats = getattr(self, 'collect_parse_stats', True)  # 是否收集解析统计信息

            # 初始化解析统计
            self.parse_stats = ParseStats.get_instance()
            if self.collect_parse_stats:
                self.parse_stats.enable()
            else:
                self.parse_stats.disable()

            # 初始化响应质量跟踪器
            self.response_quality = ResponseQualityTracker.get_instance()
            if self.collect_parse_stats:  # 使用相同的配置项
                self.response_quality.enable()
            else:
                self.response_quality.disable()

            # 初始化解析成功计数器
            self.parse_success_count = 0
            self.parse_failure_count = 0
            self.parse_retry_count = 0

            # 初始化响应质量计数器
            self.response_total_count = 0
            self.response_first_success_count = 0
            self.response_retry_success_count = 0
            self.response_failure_count = 0

            # 初始化重试相关配置
            self.llm_retry_count = getattr(self, 'max_retries', 2)  # 使用初始化时传入的max_retries参数，默认为2

        except Exception as e:
            agent_log.error(f"初始化认知知识图谱时出错: {e}")
            self.cognitive_graph = None

    def get_platform_from_channel(self):
        """
        从通信通道获取平台对象
        """
        try:
            if hasattr(self, 'twitter_channel') and hasattr(self.twitter_channel, 'platform'):
                platform = self.twitter_channel.platform
                if platform is not None:
                    # 确保环境有平台引用
                    if hasattr(self, 'env'):
                        if not hasattr(self.env, 'platform') or self.env.platform is None:
                            self.env.platform = platform
                    return platform
                else:
                    agent_log.debug("平台对象为None")
            else:
                agent_log.debug("twitter_channel不存在或没有platform属性")

            # 尝试从env中获取平台
            if hasattr(self, 'env') and hasattr(self.env, 'platform'):
                return self.env.platform

            return None
        except Exception as e:
            agent_log.error(f"获取平台对象出错: {str(e)}")
            return None



    async def _build_user_message_label(self, env_prompt, target_content=None):
        """生成CRC提示"""
        try:


            # 获取必要的提示配置
            action_space_prompt = getattr(self, 'action_space_prompt', None)
            cognition_space_dict = getattr(self, 'cognition_space_dict', None)

            # 生成提示
            total_prompt = generate_prompt(
                user=self.user_info,
                env_prompt=env_prompt,
                cognitive_profile=self.cognitive_profile,
                prompt_mode='label',  # 使用label模式
                think_mode=self.think_mode,
                action_space_prompt=action_space_prompt,
                cognition_space_dict=cognition_space_dict,
                target_context=target_content,
                time_step=self.step_counter
            )
            self.prompt = total_prompt

            # 创建单个用户消息
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=total_prompt,
            )

            # 创建最终消息列表，只包含一个用户消息
            total_massage = [user_msg.to_openai_user_message()]

            agent_log.info(f"Agent {self.agent_id} 正在运行，提示长度: {len(total_prompt)}")

            return total_massage

        except Exception as e:
            agent_log.error(f"构建用户消息时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            raise

    async def _build_user_message(self, env_prompt, save_mode="db", prompt_mode="asce"):
        """
        生成用户消息，使用模块化结构组织提示内容

        Args:
            env_prompt: 环境提示内容
            save_mode: 保存模式，"db"或"csv"，默认为"db"

        Returns:
            list: 生成的消息列表
        """
        try:

            # 获取必要的提示配置
            action_space_prompt = getattr(self, 'action_space_prompt', None)
            cognition_space_dict = getattr(self, 'cognition_space_dict', None)

            # 处理因果提示
            if self.step_counter == 1 or self.is_active == False:
                causal_prompt = self.causal_prompt
            elif self.step_counter >= 3 and self.think_mode == 'CRC-DBN' and self.step_counter % self.causal_analysis_frequency == 0:
                agent_log.info(f"Agent {self.agent_id} 第{self.step_counter}轮：生成因果关系（增强样本扩展）")
                
                # 记录因果模型计算时间开始
                causal_start_time = time.time()
                
                # 使用增强的相似用户数据获取和动态样本扩展
                causal_prompt = get_causal_relations_for_simulation(
                    user_id=self.agent_id,
                    num_steps=self.step_counter,
                    csv_path=self.csv_path,
                    method=self.causal_method,  # 使用配置的因果分析方法
                    merge=False,  # 使用增强的相似用户模式
                    merge_mode="similar",
                    top_n=20,  # 保留兼容性参数
                    filter_inactive=True,
                    enable_smart_merge=True,  # 启用智能合并机制
                    target_samples=200,  # 目标样本数量
                    min_steps_for_causal=3  # 最小轮次要求
                )
                
                # 如果增强模式仍然返回空结果且步数较早，尝试全局合并模式
                if not causal_prompt and self.step_counter <= 5:
                    agent_log.info(f"Agent {self.agent_id} 增强相似用户模式数据不足，尝试全局合并模式")
                    causal_prompt = get_causal_relations_for_simulation(
                        user_id=self.agent_id,
                        num_steps=self.step_counter,
                        csv_path=self.csv_path,
                        method=self.causal_method,
                        merge=True,
                        merge_mode="all",  # 使用全局模式作为最后手段
                        top_n=10,  # 减少用户数量以提高计算效率
                        filter_inactive=True,
                        enable_smart_merge=True,  # 启用智能合并机制
                        target_samples=400,  # 全局模式使用更大的样本数
                        min_steps_for_causal=3
                    )
                
                # 记录因果模型计算时间结束
                causal_end_time = time.time()
                causal_duration = causal_end_time - causal_start_time
                
                # 将时间记录添加到全局变量中
                if self.causal_method in causal_model_timing:
                    causal_model_timing[self.causal_method].append({
                        'agent_id': self.agent_id,
                        'step': self.step_counter,
                        'duration': causal_duration,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                # 获取并记录因果分析的数值结果
                try:
                    results_key = f"{self.causal_method}_numerical_results"
                    if hasattr(get_causal_relations_for_simulation, results_key):
                        results_list = getattr(get_causal_relations_for_simulation, results_key)
                        if results_list and self.causal_method in causal_model_results:
                            # 获取最新的结果（刚刚计算的）
                            latest_result = results_list[-1]
                            causal_model_results[self.causal_method].append(latest_result)
                            agent_log.info(f"Agent {self.agent_id} 因果数值结果已记录，矩阵维度: {len(latest_result.get('causal_matrix', []))}x{len(latest_result.get('causal_matrix', [[]]))} ")
                except Exception as e:
                    agent_log.warning(f"Agent {self.agent_id} 记录因果数值结果时出错: {e}")
                
                agent_log.info(f"Agent {self.agent_id} 因果分析({self.causal_method})耗时: {causal_duration:.4f}秒")
                self.causal_prompt = causal_prompt
            else:
                causal_prompt = self.causal_prompt

            # 获取历史记忆
            self.history_memory = await self.generate_memory_content_csv(self.step_counter)

            # 检查环境提示长度并截断过长内容
            if len(env_prompt) > 40000:
                env_prompt = env_prompt[:40000]
                agent_log.warning(f"Agent {self.agent_id} 环境提示过长，已截断至40000字符")
            #print(json.dumps({"env_prompt": env_prompt}, ensure_ascii=False))
            # 生成提示
            total_prompt = generate_prompt(
                user=self.user_info,
                env_prompt=env_prompt,
                cognitive_profile=self.cognitive_profile,
                prompt_mode=prompt_mode,
                think_mode=self.think_mode,
                action_space_prompt=action_space_prompt,
                cognition_space_dict=cognition_space_dict,
                causal_prompt=self.causal_prompt,
                memory_content=self.history_memory,
                time_step=self.step_counter
            )

            # 检查总提示长度
            if len(total_prompt) > 80000:
                agent_log.warning(f"Agent {self.agent_id} 总提示过长，将进一步截断环境提示")
                env_prompt = env_prompt[:40000]
                total_prompt = generate_prompt(
                    user=self.user_info,
                    env_prompt=env_prompt,
                    cognitive_profile=self.cognitive_profile,
                    prompt_mode=prompt_mode,
                    think_mode=self.think_mode,
                    action_space_prompt=action_space_prompt,
                    cognition_space_dict=cognition_space_dict,
                    causal_prompt=self.causal_prompt,
                    memory_content=self.history_memory,
                    time_step=self.step_counter
                )

            agent_log.info(f"Agent {self.agent_id} 第{self.step_counter}轮：提示长度={len(total_prompt)}，上下文长度={len(env_prompt)}")

            # 创建用户消息
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=total_prompt,
            )

            # 创建最终消息列表
            total_massage = [user_msg.to_openai_user_message()]

            return total_massage

        except Exception as e:
            agent_log.error(f"构建用户消息时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            raise

    async def _validate_messages(self, openai_messages):
        """验证消息格式"""
        try:
            valid_messages = []
            invalid_count = 0

            for i, msg in enumerate(openai_messages):
                is_valid = True

                # 验证消息格式
                if not isinstance(msg, dict):
                    invalid_count += 1
                    is_valid = False
                    continue

                if "role" not in msg:
                    invalid_count += 1
                    is_valid = False
                    continue

                # 处理content字段
                if "content" not in msg:
                    msg["content"] = ""
                elif msg["content"] is None:
                    msg["content"] = ""
                elif not isinstance(msg["content"], str):
                    msg["content"] = str(msg["content"])

                if is_valid:
                    valid_messages.append(msg)

            # 只记录关键信息
            if invalid_count > 0:
                agent_log.warning(f"Agent {self.agent_id}：消息验证 - 发现{invalid_count}条无效消息")

            if not valid_messages:
                agent_log.error(f"Agent {self.agent_id}：消息验证 - 没有有效的消息")
                return None

            return valid_messages

        except Exception as e:
            agent_log.error(f"Agent {self.agent_id}：验证消息时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return None

    async def _prepare_deepseek_messages(self, valid_messages):
        """为DeepSeek模型准备消息"""
        for i, msg in enumerate(valid_messages):
            # 确保role字段是DeepSeek支持的值
            if msg["role"] not in ["user", "assistant", "system"]:
                original_role = msg["role"]
                if msg["role"].lower() == "user":
                    valid_messages[i]["role"] = "user"
                elif msg["role"].lower() == "assistant":
                    valid_messages[i]["role"] = "assistant"
                else:
                    valid_messages[i]["role"] = "user"  # 默认设为user
                agent_log.warning(f"警告: 消息的role字段'{original_role}'不被DeepSeek支持，已转换为'{valid_messages[i]['role']}'")

        return valid_messages

    async def _call_model(self, valid_messages):
        """调用模型并获取响应"""
        try:
            agent_log.info(f"Agent {self.agent_id} 正在调用模型")

            if self.is_local_model:
                # 如果是本地模型，使用专用的API调用函数
                response = await self._call_local_model_api(valid_messages)
            else:
                # 否则使用标准的模型后端
                response = self.model_backend.run(valid_messages)

            agent_log.info(f"Agent {self.agent_id} 成功获取模型响应")
            return response
        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 调用模型失败: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return {"error": str(e)}

    async def _extract_reason(self, content):
        """从内容中提取理由"""
        try:
            reason = ""
            reasoning_line_idx = -1

            # 定义匹配模式
            patterns = [
                (r"Reasoning: (.+?)(?=\n|$)", "英文Reasoning"),
                (r"Reason: (.+?)(?=\n|$)", "英文Reason"),
                (r"原因: (.+?)(?=\n|$)", "中文原因"),
                (r"意图: (.+?)(?=\n|$)", "中文意图")
            ]

            # 尝试多种模式匹配
            for pattern, pattern_name in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    reason = match.group(1).strip()
                    break

            # 如果找不到特定模式，尝试直接提取语义上的理由
            if not reason:
                lines = content.lower().split('\n')
                # 寻找包含关键词的行
                keywords = ["reason", "reasoning", "原因", "理由", "意图", "because", "motivation"]
                for i, line in enumerate(lines):
                    if any(keyword in line for keyword in keywords):
                        reasoning_line_idx = i
                        if ":" in line:
                            reason = line.split(":", 1)[1].strip()
                        break

                # 如果找到了相关行但没有提取到理由，尝试使用下一行
                if reasoning_line_idx >= 0 and not reason and reasoning_line_idx + 1 < len(lines):
                    reason = lines[reasoning_line_idx + 1].strip()

            # 如果仍然没有理由，尝试从content中提取摘要
            if not reason and len(content) > 0:
                reason = content[:50] + ("..." if len(content) > 50 else "")

            # 过滤特殊字符和过长的理由
            if reason:
                # 去除特殊字符
                reason = re.sub(r'[^\w\s,.?!;:\-\'"]', '', reason).strip()
                # 限制长度
                reason = reason[:100] + ("..." if len(reason) > 100 else "")

            # 如果提取的理由为空，设置默认值
            if not reason:
                reason = "No reason provided"
                agent_log.warning(f"Agent {self.agent_id} 未能提取到行动理由，使用默认值")

            # 更新实例的reason属性
            self.reason = reason

            return reason
        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 提取理由时出错: {str(e)}")
            self.reason = "Error extracting reason"
            return "Error extracting reason"
    def _extract_cognitive_state(self, content, cognitive_state=None):
        """
        从LLM响应中提取认知状态

        具有强鲁棒性和自愈能力，确保即使LLM输出不规范，也能尽可能提取有效信息

        参数:
            content: LLM的响应内容
            cognitive_state: 初始认知状态，如果提供则在此基础上更新
        返回:
            提取后的认知状态字典，格式与DEFAULT_COGNITIVE_PROFILE一致
        """
        import copy
        import json

        # 如果没有提供初始认知状态，则使用默认配置，并深拷贝以避免修改原始数据
        if cognitive_state is None:
            cognitive_state = copy.deepcopy(self.DEFAULT_COGNITIVE_PROFILE)
        else:
            cognitive_state = copy.deepcopy(cognitive_state)

        # 在解析前复制一份当前认知状态的备份，用于在解析出现问题或内容缺失时恢复已有值
        cognitive_profile_copy = copy.deepcopy(cognitive_state)

        agent_log.info(f"Agent {self.agent_id} 初始认知状态: {cognitive_state}")

        # 如果内容为空，则直接返回初始认知状态
        if not content:
            agent_log.warning("提供的内容为空，返回初始认知状态")
            return cognitive_state

        # 1. 预处理：修复和清洗JSON格式
        # 替换双花括号为单花括号，移除```json```标记，清理额外空格
        try:
            content_fixed = content.replace('{{', '{').replace('}}', '}')
            content_fixed = re.sub(r'```(?:json)?|```', '', content_fixed).strip()
        except Exception as e:
            agent_log.error(f"预处理内容时出错: {e}")
            content_fixed = content  # 如果预处理失败，使用原始内容

        extracted_data = None
        missing_fields = []

        # 2. 多种方式尝试提取有效JSON
        try:
            # 2.1 直接尝试解析整个内容作为JSON
            try:
                extracted_data = json.loads(content_fixed)
                agent_log.debug("直接解析整个内容成功")
            except json.JSONDecodeError:
                pass

            # 2.2 使用正则表达式查找JSON代码块
            if not extracted_data:
                json_match = re.search(r'{[\s\S]*?}', content_fixed, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                    try:
                        extracted_data = json.loads(json_str)
                        agent_log.debug("通过正则提取JSON成功")
                    except json.JSONDecodeError as e:
                        agent_log.warning(f"JSON正则提取解析错误: {e}")

            # 2.3 手动查找并匹配大括号对
            if not extracted_data:
                agent_log.debug("尝试手动匹配大括号...")
                start_idx = content_fixed.find('{')
                if start_idx >= 0:
                    # 尝试找到匹配的结束花括号
                    open_count = 1
                    for i in range(start_idx + 1, len(content_fixed)):
                        if content_fixed[i] == '{':
                            open_count += 1
                        elif content_fixed[i] == '}':
                            open_count -= 1
                            if open_count == 0:
                                # 找到了匹配的结束括号
                                json_str = content_fixed[start_idx:i+1]
                                try:
                                    extracted_data = json.loads(json_str)
                                    agent_log.debug(f"手动匹配括号提取JSON成功")
                                    break
                                except json.JSONDecodeError as e:
                                    agent_log.warning(f"尝试解析手动提取的JSON出错: {e}")
                                    # 继续尝试查找下一个闭括号

                # 2.4 如果所有方法都失败，尝试修复并再次解析
                if not extracted_data and start_idx >= 0:
                    # 查找最后一个大括号
                    end_idx = content_fixed.rfind('}')
                    if end_idx > start_idx:
                        json_str = content_fixed[start_idx:end_idx+1]
                        # 尝试修复常见格式问题
                        json_str = re.sub(r',\s*}', '}', json_str)  # 移除JSON对象末尾多余的逗号
                        json_str = re.sub(r',\s*]', ']', json_str)  # 移除数组末尾多余的逗号
                        try:
                            extracted_data = json.loads(json_str)
                            agent_log.debug("通过修复格式提取JSON成功")
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            agent_log.error(f"提取JSON时出错: {e}")
        # 3. 如果所有方法都失败，返回初始认知状态，但不抛出异常
        if not extracted_data:
            agent_log.warning("无法提取有效JSON数据，返回初始认知状态")
            return cognitive_state

        # 4. 提取认知状态数据
        # 4.1 检查是否有cognitive_state子对象
        cognitive_data = None
        if "cognitive_state" in extracted_data and isinstance(extracted_data["cognitive_state"], dict):
            cognitive_data = extracted_data["cognitive_state"]
            agent_log.debug(f"提取到cognitive_state子对象")
        else:
            # 4.2 如果没有cognitive_state子对象，尝试从顶层对象提取认知数据
            cognitive_data = extracted_data
            agent_log.debug("从顶层对象提取认知数据")

        # 5. 提取并更新认知状态的各个字段
        # 5.1 处理基本认知字段 - 保留已有字段，即使部分缺失
        for field in ["mood", "emotion", "stance", "thinking", "intention"]:
            field_updated = False

            # 字段存在且为字典格式
            if field in cognitive_data and isinstance(cognitive_data[field], dict):
                field_data = cognitive_data[field]

                # 更新类型
                if "type" in field_data and field_data["type"]:
                    normalized_type = self._normalize_cognitive_type(field, field_data["type"])
                    cognitive_state[field]["type"] = normalized_type
                    field_updated = True

                # 更新值
                if "value" in field_data and field_data["value"]:
                    normalized_value = self._normalize_cognitive_value(
                        field,
                        cognitive_state[field]["type"],
                        field_data["value"]
                    )
                    cognitive_state[field]["value"] = normalized_value
                    field_updated = True

            # 字段为简单字符串格式
            elif field in cognitive_data and isinstance(cognitive_data[field], str):
                # 直接使用字符串作为type，值保持不变
                normalized_type = self._normalize_cognitive_type(field, cognitive_data[field])
                cognitive_state[field]["type"] = normalized_type
                field_updated = True

            if not field_updated:
                missing_fields.append(field)
                agent_log.warning(f"字段 {field} 缺失或格式不正确，保留原始值")

        # 6. 处理观点(opinion)数据 - 增强容错和补全
        # 确保认知状态中存在opinion键，初始值使用备份中的数据
        if "opinion" not in cognitive_state or not isinstance(cognitive_state["opinion"], dict):
            cognitive_state["opinion"] = copy.deepcopy(cognitive_profile_copy.get("opinion", {}))

        # 6.1 尝试提取opinion数据
        opinion_updated = False
        if "opinion" in cognitive_data:
            opinion_data = cognitive_data["opinion"]
            opinion_dict = {}

            # 6.2 处理列表格式的opinion
            if isinstance(opinion_data, list):
                for item in opinion_data:
                    if isinstance(item, dict):
                        # 处理观点格式: {"viewpoint_1": "支持级别"}
                        for key, value in item.items():
                            if key.startswith("viewpoint_"):
                                # 直接使用值作为支持级别
                                normalized_support = self._normalize_support_level(value)
                                opinion_dict[key] = normalized_support
                                opinion_updated = True
                                agent_log.debug(f"从列表更新 {key}: {normalized_support}")

            # 6.3 处理字典格式的opinion
            elif isinstance(opinion_data, dict):
                # 直接格式: {"viewpoint_1": "support_level", ...}
                for key, value in opinion_data.items():
                    if key.startswith("viewpoint_"):
                        normalized_support = self._normalize_support_level(value)
                        opinion_dict[key] = normalized_support
                        opinion_updated = True
                        agent_log.debug(f"从字典更新 {key}: {normalized_support}")

            # 6.4 合并已提取的观点数据
            if opinion_dict:
                for key, value in opinion_dict.items():
                    cognitive_state["opinion"][key] = value

        # 确保最终的opinion是字典格式，而不是列表格式
        if isinstance(cognitive_state["opinion"], list):
            agent_log.info("将opinion从列表格式转换为字典格式")
            opinion_dict = {}
            for item in cognitive_state["opinion"]:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key.startswith("viewpoint_"):
                            opinion_dict[key] = self._normalize_support_level(value)
            cognitive_state["opinion"] = opinion_dict
            agent_log.debug(f"转换后的opinion字典: {opinion_dict}")

        # 7. 确保所有六个viewpoint键存在，补全缺失项
        viewpoint_count = 0
        for i in range(1, 7):
            key = f"viewpoint_{i}"
            if key in cognitive_state["opinion"]:
                viewpoint_count += 1
            else:
                # 使用默认值或备份中的值
                cognitive_state["opinion"][key] = cognitive_profile_copy["opinion"].get(key, "Indifferent")
                agent_log.debug(f"补全缺失的 {key}: {cognitive_state['opinion'][key]}")

        # 8. 记录提取结果统计
        if missing_fields:
            agent_log.warning(f"以下字段缺失或格式错误: {', '.join(missing_fields)}")

        if viewpoint_count < 6:
            agent_log.warning(f"补全了 {6 - viewpoint_count} 个缺失的观点")

        agent_log.debug(f"最终认知状态: {cognitive_state}")
        return cognitive_state

    def _normalize_support_level(self, support_level, valid_support_levels=None):
        """规范化观点支持级别，使用NORMALIZATION_MAP中的映射规则

        具有增强的鲁棒性，可以处理各种格式的支持级别表达方式

        Args:
            support_level: 原始支持级别，可以是字符串、数字或其他类型
            valid_support_levels: 有效的支持级别列表

        Returns:
            规范化后的支持级别，确保返回一个有效的字符串
        """
        # 处理特殊情况
        if support_level is None:
            return "Indifferent"

        # 尝试将支持级别转换为字符串
        try:
            support_level = str(support_level).strip()
        except Exception as e:
            agent_log.warning(f"转换支持级别为字符串时出错: {e}")
            return "Indifferent"

        # 如果支持级别为空字符串，返回默认值
        if not support_level:
            return "Indifferent"

        # 如果没有提供有效支持级别列表，使用默认列表
        if not valid_support_levels:
            # 尝试从认知空间字典获取有效支持级别
            if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict and 'opinion_support_levels' in self.cognition_space_dict:
                valid_support_levels = self.cognition_space_dict['opinion_support_levels']
            else:
                # 使用默认列表
                valid_support_levels = [
                'Strongly Support', 'Moderate Support', 'Do Not Support',
                'Moderate Opposition', 'Strongly Opposition', 'Indifferent']

        # 转换为小写以进行不区分大小写的匹配
        support_lower = support_level.lower()

        # 检查全局NORMALIZATION_MAP是否有支持级别映射
        if NORMALIZATION_MAP and 'support_levels' in NORMALIZATION_MAP:
            # 遍历所有标准支持级别
            for standard_level, variants in NORMALIZATION_MAP['support_levels'].items():
                # 检查输入是否匹配任何变体
                if any(variant in support_lower or support_lower in variant for variant in variants):
                    # 确保标准支持级别在有效支持级别列表中
                    if standard_level in valid_support_levels:
                        return standard_level
                    else:
                        agent_log.warning(f"规范化支持级别 '{standard_level}' 不在有效支持级别列表中")
                        # 如果不在有效列表中，返回最接近的有效支持级别
                        if "Strong" in standard_level and "Support" in standard_level:
                            for level in valid_support_levels:
                                if "Strong" in level and "Support" in level:
                                    return level
                        elif "Moderate" in standard_level and "Support" in standard_level:
                            for level in valid_support_levels:
                                if "Moderate" in level and "Support" in level:
                                    return level
                        elif "Not" in standard_level or "Do Not" in standard_level:
                            for level in valid_support_levels:
                                if "Not" in level or "Do Not" in level:
                                    return level
                        elif "Opposition" in standard_level:
                            for level in valid_support_levels:
                                if "Opposition" in level:
                                    return level
                        elif "Indifferent" in standard_level or "Neutral" in standard_level:
                            for level in valid_support_levels:
                                if "Indifferent" in level or "Neutral" in level:
                                    return level

                        # 如果没有找到合适的匹配，返回第一个有效支持级别
                        if valid_support_levels:
                            return valid_support_levels[0]

        # 1. 精确匹配 - 检查是否与有效支持级别完全匹配（忽略大小写）
        for level in valid_support_levels:
            if support_lower == level.lower():
                        return level

        # 2. 部分匹配 - 检查支持级别是否包含有效支持级别（或反之）
        for level in valid_support_levels:
            if level.lower() in support_lower or support_lower in level.lower():
                return level


        # 4. 关键词匹配 - 查找支持/反对/中立等关键词
        support_keywords = ['support', 'agree', 'favor', 'approve', 'endorse', 'back', 'positive']
        oppose_keywords = ['opposition', 'disagree', 'against', 'reject', 'disapprove', 'negative']
        neutral_keywords = ['neutral', 'indifferent', 'ambivalent', 'undecided', 'middle']

        # 强度修饰词
        strong_modifiers = ['strong', 'very', 'extremely', 'fully', 'completely', 'highly', 'totally']
        moderate_modifiers = ['moderate', 'somewhat', 'partially', 'slight', 'mild', 'limited']

        # 检查支持关键词
        if any(keyword in support_lower for keyword in support_keywords):
            # 检查强度修饰词
            if any(modifier in support_lower for modifier in strong_modifiers):
                return 'Strongly Support'
            elif any(modifier in support_lower for modifier in moderate_modifiers):
                return 'Moderate Support'
            return 'Moderate Support'  # 默认为中度支持

        # 检查反对关键词
        if any(keyword in support_lower for keyword in oppose_keywords):
            # 检查强度修饰词
            if any(modifier in support_lower for modifier in strong_modifiers):
                return 'Strongly Opposition'
            elif any(modifier in support_lower for modifier in moderate_modifiers):
                return 'Moderate Opposition'
            return 'Do Not Support'  # 默认为不支持

        # 检查中立关键词
        if any(keyword in support_lower for keyword in neutral_keywords):
            return 'Indifferent'

        # 5. 词组分析 - 分析支持级别中的词组
        if 'not' in support_lower or 'no' in support_lower.split() or 'don\'t' in support_lower:
            if any(keyword in support_lower for keyword in support_keywords):
                return 'Do Not Support'

        # 6. 默认策略 - 如果所有匹配都失败，返回默认值
        agent_log.warning(f"无法确定支持级别 '{support_level}' 的含义，使用默认值 'Indifferent'")
        return 'Indifferent'

    def _normalize_cognitive_type(self, field, cog_type):
        """规范化认知类型，使用cognitive_normalization.json中的映射"""
        # 默认类型
        default_type = 'neutral'
        if field == 'emotion':
            default_type = 'complex'  # emotion默认用complex
        elif field == 'thinking':
            default_type = 'analytical'  # thinking默认用analytical
        elif field == 'intention':
            default_type = 'expressive'  # intention默认用expressive

        # 如果类型为空，返回默认类型
        if not cog_type:
            return default_type

        # 尝试从NORMALIZATION_MAP中获取类型映射
        field_type_key = f"{field}_type"
        if NORMALIZATION_MAP and field_type_key in NORMALIZATION_MAP:
            # 将输入类型转换为小写用于匹配
            cog_type_lower = cog_type.lower()

            # 遍历标准类型及其变体
            for standard_type, variants in NORMALIZATION_MAP[field_type_key].items():
                if any(variant == cog_type_lower or variant in cog_type_lower or cog_type_lower in variant for variant in variants):
                    # 找到匹配的标准类型
                    return standard_type

        # 从认知空间字典获取有效类型列表
        valid_field_types = {}
        if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict:
            for f in ['mood', 'emotion', 'thinking', 'stance', 'intention']:
                if f in self.cognition_space_dict and 'type_list' in self.cognition_space_dict[f]:
                    valid_field_types[f] = self.cognition_space_dict[f]['type_list']

        # 如果认知空间字典中缺少某些字段，使用默认值
        if 'mood' not in valid_field_types:
            valid_field_types['mood'] = ['positive', 'negative', 'neutral', 'ambivalent']
        if 'emotion' not in valid_field_types:
            valid_field_types['emotion'] = ['positive', 'negative', 'complex']
        if 'stance' not in valid_field_types:
            valid_field_types['stance'] = ['conservative', 'radical', 'neutral']
        if 'thinking' not in valid_field_types:
            valid_field_types['thinking'] = ['intuitive', 'analytical', 'authority_dependent', 'critical']
        if 'intention' not in valid_field_types:
            valid_field_types['intention'] = ['expressive', 'active', 'observant', 'resistant']

        # 检查类型是否有效
        if field in valid_field_types and cog_type in valid_field_types[field]:
            return cog_type

        # 如果类型无效，尝试查找最相似的类型
        if field in valid_field_types:
            valid_types = valid_field_types[field]

            # 去除重复的小写版本
            unique_types = []
            for t in valid_types:
                if t.lower() not in [ut.lower() for ut in unique_types]:
                    unique_types.append(t)

            # 尝试模糊匹配
            for valid_type in unique_types:
                if valid_type.lower() in cog_type.lower() or cog_type.lower() in valid_type.lower():
                    return valid_type

            # 尝试部分单词匹配
            cog_type_words = cog_type.lower().split()
            for word in cog_type_words:
                if len(word) > 3:  # 只考虑长度超过3的单词，避免匹配太短的词
                    for valid_type in unique_types:
                        if word in valid_type.lower() or any(word in vt.lower() for vt in valid_type.split()):
                            return valid_type

            # 如果找不到匹配，使用第一个有效类型
            if unique_types:
                return unique_types[0]

        # 如果找不到匹配，返回默认类型
        return default_type

    def _normalize_cognitive_value(self, field, cog_type, cog_value):
        """规范化认知值，使用cognitive_normalization.json中的映射"""
        # 如果值为空，使用类型作为值
        if not cog_value:
            return cog_type

        # 处理多值情况（以逗号、and、or或其他分隔符分隔的多个值）
        if ',' in cog_value or ' and ' in cog_value or '/' in cog_value or ' or ' in cog_value:
            # 提取可能的多个值
            value_parts = re.split(r',|\s+and\s+|/|\s+or\s+', cog_value)
            value_parts = [part.strip().lower() for part in value_parts if part.strip()]

            # 只保留第一个值
            if value_parts:
                cog_value = value_parts[0]

        # 尝试从NORMALIZATION_MAP中获取值映射
        field_value_key = f"{field}_value"
        if NORMALIZATION_MAP and field_value_key in NORMALIZATION_MAP:
            # 将输入值转换为小写用于匹配
            cog_value_lower = cog_value.lower()

            # 尝试直接匹配标准值
            for standard_value, variants in NORMALIZATION_MAP[field_value_key].items():
                if any(variant == cog_value_lower or variant in cog_value_lower or cog_value_lower in variant for variant in variants):
                    # 找到匹配的标准值
                    return standard_value

        # 从认知空间字典获取有效值列表
        valid_field_values = {}
        if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict:
            for f in ['mood', 'emotion', 'thinking', 'stance', 'intention']:
                if f in self.cognition_space_dict and 'value_list' in self.cognition_space_dict[f]:
                    valid_field_values[f] = self.cognition_space_dict[f]['value_list']

        # 如果认知空间字典中缺少某些字段，使用默认值
        if 'mood' not in valid_field_values:
            valid_field_values['mood'] = ['Optimistic', 'Confident', 'Passionate', 'Empathetic', 'Grateful',
                                             'Pessimistic', 'Apathetic', 'Distrustful', 'Realistic', 'Rational',
                                             'Prudent', 'Detached', 'Objective']
        if 'emotion' not in valid_field_values:
            valid_field_values['emotion'] = ['Excited', 'Satisfied', 'Joyful', 'Touched', 'Calm',
                                           'Angry', 'Anxious', 'Depressed', 'Fearful', 'Disgusted',
                                           'Conflicted', 'Doubtful', 'Hesitant', 'Surprised', 'Helpless']
        if 'thinking' not in valid_field_values:
            valid_field_values['thinking'] = ['Subjective', 'Gut Feeling', 'Experience-based',
                                             'Logical', 'Evidence-based', 'Data-driven',
                                             'Follow Mainstream', 'Trust Experts', 'Obey Authority',
                                             'Skeptical', 'Questioning', 'Identifying Flaws']
        if 'stance' not in valid_field_values:
            valid_field_values['stance'] = ['Respect Authority', 'Emphasize Stability', 'Preserve Traditions',
                                          'Challenge Authority', 'Break Conventions', 'Promote Change',
                                          'Compromise', 'Balance Perspectives', 'Pragmatic']
        if 'intention' not in valid_field_values:
            valid_field_values['intention'] = ['Commenting', 'Writing Articles', 'Joining Discussions',
                                             'Organizing Events', 'Advocating Actions', 'Voting',
                                             'Observing', 'Recording', 'Remaining Silent',
                                             'Opposing', 'Arguing', 'Protesting']

        # 检查值是否有效
        if field in valid_field_values:
            valid_values = [v.lower() for v in valid_field_values[field]]

            # 如果值在有效列表中，返回原始值
            if cog_value.lower() in valid_values:
                for v in valid_field_values[field]:
                    if v.lower() == cog_value.lower():
                        return v

            # 尝试模糊匹配
            for valid_value in valid_field_values[field]:
                if valid_value.lower() in cog_value.lower() or cog_value.lower() in valid_value.lower():
                    return valid_value

            # 尝试部分单词匹配
            cog_value_words = cog_value.lower().split()
            for word in cog_value_words:
                if len(word) > 3:  # 只考虑长度超过3的单词，避免匹配太短的词
                    for valid_value in valid_field_values[field]:
                        valid_value_words = valid_value.split()
                        if word in valid_value.lower() or any(word in vw.lower() for vw in valid_value_words):
                            return valid_value

            # 如果找不到匹配，返回该类型的第一个有效值
            if valid_field_values[field]:
                return valid_field_values[field][0]

        # 如果找不到有效值，返回类型作为值
        return cog_value

    def _fallback_llm_parse(self, content, default_state):
        """当正则匹配失败时，使用LLM辅助解析"""
        prompt = f"""请从以下文本中提取认知状态，按JSON格式返回：
        {content[:1000]}

        要求包含以下字段（均为小写英文）：
        - mood (positive/negative/neutral)
        - emotion (happy/angry/sad/neutral)
        - stance (support/opposition/neutral)
        - thinking (aware/confused/neutral)
        - intention (interested/uninterested/neutral)"""
        try:
            response = self.model_backend.run([{"role": "user", "content": prompt}])
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"备用解析失败，使用默认状态: {str(e)}")
            return default_state

    async def _handle_function_calls(self, functions, content):
        """统一处理函数调用逻辑"""
        try:
            func_info = functions[0]
            called_name = func_info.get("name", "") or func_info.get("action", "")
            args = func_info.get("arguments", {}) or func_info.get("params", {})

            # 执行动作并更新记录
            result = await self._execute_action(called_name, args)
            return content[:1000]
        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 函数调用处理失败: {str(e)}")
            return content

    def check_cognitive_profile(self):

        if hasattr(self, "users_cognitive_profile_dict"):
            if self.agent_id in self.users_cognitive_profile_dict:
                backup_profile = self.users_cognitive_profile_dict[self.agent_id]

                if ('mood' in self.cognitive_profile and (self.cognitive_profile['mood'].get('type') == 'fail' or
                    self.cognitive_profile['mood'].get('value') == 'fail')):
                    if 'mood' in backup_profile:
                        self.cognitive_profile['mood'] = backup_profile['mood'].copy()
                        agent_log.info(f"用户{self.agent_id}的情感维度从备份中恢复")

                if ('emotion' in self.cognitive_profile and
                    (self.cognitive_profile['emotion'].get('type') == 'fail' or
                    self.cognitive_profile['emotion'].get('value') == 'fail')):
                        if 'emotion' in backup_profile:
                            self.cognitive_profile['emotion'] = backup_profile['emotion'].copy()
                            agent_log.info(f"用户{self.agent_id}的情绪维度从备份中恢复")

                if ('thinking' in self.cognitive_profile and
                    (self.cognitive_profile['thinking'].get('type') == 'fail' or
                    self.cognitive_profile['thinking'].get('value') == 'fail')):
                    if 'thinking' in backup_profile:
                            self.cognitive_profile['thinking'] = backup_profile['thinking'].copy()
                            agent_log.info(f"用户{self.agent_id}的认知维度从备份中恢复")

                if ('stance' in self.cognitive_profile and
                    (self.cognitive_profile['stance'].get('type') == 'fail' or
                    self.cognitive_profile['stance'].get('value') == 'fail')):
                        if 'stance' in backup_profile:
                            self.cognitive_profile['stance'] = backup_profile['stance'].copy()
                            agent_log.info(f"用户{self.agent_id}的立场维度从备份中恢复")

                if ('intention' in self.cognitive_profile and
                    (self.cognitive_profile['intention'].get('type') == 'fail' or
                    self.cognitive_profile['intention'].get('value') == 'fail')):
                        if 'intention' in backup_profile:
                            self.cognitive_profile['intention'] = backup_profile['intention'].copy()
                            agent_log.info(f"用户{self.agent_id}的意图维度从备份中恢复")

                if 'opinion' in self.cognitive_profile:
                    if not isinstance(self.cognitive_profile['opinion'], list) and not isinstance(self.cognitive_profile['opinion'], dict):
                        if 'opinion' in backup_profile:
                            self.cognitive_profile['opinion'] = backup_profile['opinion']
                            agent_log.info(f"用户{self.agent_id}的整个观点维度从备份中恢复")
                    else:
                        # 将列表格式的opinion转换为字典格式，保持一致性
                        if isinstance(self.cognitive_profile['opinion'], list):
                            agent_log.info(f"将用户{self.agent_id}的opinion从列表格式转换为字典格式")
                            opinion_dict = {}
                            for item in self.cognitive_profile['opinion']:
                                if isinstance(item, dict):
                                    for key, value in item.items():
                                        if key.startswith('viewpoint_'):
                                            opinion_dict[key] = self._normalize_support_level(value)
                            # 只有当成功提取了观点数据时才替换
                            if opinion_dict:
                                self.cognitive_profile['opinion'] = opinion_dict
                                agent_log.debug(f"转换后的opinion字典: {opinion_dict}")
                            else:
                                agent_log.warning(f"无法从列表中提取有效的观点数据")
                                # 尝试从备份恢复
                                if 'opinion' in backup_profile and isinstance(backup_profile['opinion'], dict):
                                    self.cognitive_profile['opinion'] = backup_profile['opinion'].copy()
                                    agent_log.info(f"用户{self.agent_id}的观点从备份中恢复")

                        # 确保字典格式的opinion包含所有所需的viewpoint键
                        if isinstance(self.cognitive_profile['opinion'], dict):
                            for viewpoint_key in ['viewpoint_1', 'viewpoint_2', 'viewpoint_3', 'viewpoint_4', 'viewpoint_5', 'viewpoint_6']:
                                if viewpoint_key not in self.cognitive_profile['opinion'] or self.cognitive_profile['opinion'][viewpoint_key] == 'fail':
                                    if 'opinion' in backup_profile and isinstance(backup_profile['opinion'], dict) and viewpoint_key in backup_profile['opinion']:
                                        self.cognitive_profile['opinion'][viewpoint_key] = backup_profile['opinion'][viewpoint_key]
                                        agent_log.info(f"用户{self.agent_id}的观点{viewpoint_key}从备份中恢复")
                                    else:
                                        # 如果备份中也没有，设置默认值
                                        self.cognitive_profile['opinion'][viewpoint_key] = "Indifferent"
                                        agent_log.info(f"用户{self.agent_id}的观点{viewpoint_key}设置为默认值'Indifferent'")

                        # 以下列表格式处理逻辑保留，但由于已经进行转换，应该不会执行
                        elif isinstance(self.cognitive_profile['opinion'], list):
                            # 检查备份配置文件是否有效
                            if not ('opinion' in backup_profile and isinstance(backup_profile['opinion'], list)):
                                agent_log.warning(f"用户{self.agent_id}的备份配置文件中opinion字段无效或不是列表类型")
                                return

                            for i, opinion_item in enumerate(self.cognitive_profile['opinion']):
                                if not isinstance(opinion_item, dict):
                                    agent_log.warning(f"用户{self.agent_id}的opinion列表中第{i}项不是字典类型")
                                    continue

                                # 检查备份配置文件中是否有对应索引的项
                                if i >= len(backup_profile['opinion']):
                                    agent_log.warning(f"用户{self.agent_id}的备份配置文件中opinion列表长度不足,缺少索引{i}")
                                    continue

                                # 恢复viewpoint字段
                                for viewpoint_key in opinion_item:
                                    if viewpoint_key.startswith('viewpoint_') and opinion_item[viewpoint_key] == 'fail':
                                        try:
                                            if viewpoint_key in backup_profile['opinion'][i]:
                                                self.cognitive_profile['opinion'][i][viewpoint_key] = backup_profile['opinion'][i][viewpoint_key]
                                                agent_log.info(f"用户{self.agent_id}的列表观点{viewpoint_key}从备份中恢复")
                                            else:
                                                agent_log.warning(f"用户{self.agent_id}的备份配置文件中缺少{viewpoint_key}")
                                        except Exception as e:
                                            agent_log.error(f"恢复viewpoint时出错: {str(e)}")

                                # 在新的格式中不再需要单独恢复type_support_levels

    async def update_num_steps(self, timestep: int):
        self.step_counter = timestep
        agent_log.debug(f"已更新所有智能体的轮次数: {self.step_counter}")

    async def generate_memory_content(self, current_timestep: int) -> str:
        """
        生成智能体的记忆内容，包括最近5轮的用户行为和认知状态

        Args:
            current_timestep: 当前时间步

        Returns:
            str: 记忆内容文本
        """
        try:
            # 获取平台实例
            platform = self.get_platform_from_channel()
            if not platform:
                agent_log.warning(f"Agent {self.agent_id} 无法获取平台实例，无法获取用户行为记录")
                return "无法获取历史记忆。"

            # 计算开始轮次（最多获取前2轮）
            start_timestep = max(1, current_timestep - 2)

            # 查询用户行为数据
            query = """
            SELECT ua.num_steps, ua.action, ua.reason, ua.post_id, ua.is_active,
                   ua.mood_type, ua.mood_value,
                   ua.emotion_type, ua.emotion_value,
                   ua.stance_type, ua.stance_value,
                   ua.thinking_type, ua.thinking_value,
                   ua.intention_type, ua.intention_value,
                   ua.viewpoint_1, ua.viewpoint_2, ua.viewpoint_3,
                   ua.viewpoint_4, ua.viewpoint_5, ua.viewpoint_6,
                   p.content as post_content
            FROM user_action ua
            LEFT JOIN post p ON ua.post_id = p.post_id
            WHERE ua.user_id = ? AND ua.num_steps >= ? AND ua.num_steps < ?
            ORDER BY ua.num_steps ASC
            """

            platform.pl_utils._execute_db_command(query, (self.agent_id, start_timestep, current_timestep))
            rows = platform.pl_utils.db_cursor.fetchall()

            if current_timestep==1 and not rows:
               # agent_log.warning(f"Step {current_timestep} No behavior records found for user {self.agent_id} between steps {start_timestep} and {current_timestep-1}")
                return "No historical memory."

            # Build memory content
            memory_content = f"### Historical Memory for User {self.agent_id} (Steps {start_timestep} to {current_timestep-1})\n\n"

            for row in rows:
                timestep = row[0]
                action = row[1]
                reason = row[2]
                post_id = row[3]
                is_active = row[4] == "true"

                # 认知状态
                mood_type = row[5]
                mood_value = row[6]
                emotion_type = row[7]
                emotion_value = row[8]
                stance_type = row[9]
                stance_value = row[10]
                thinking_type = row[11]
                thinking_value = row[12]
                intention_type = row[13]
                intention_value = row[14]

                # 观点支持级别
                viewpoints = {
                    "viewpoint_1": row[15],
                    "viewpoint_2": row[16],
                    "viewpoint_3": row[17],
                    "viewpoint_4": row[18],
                    "viewpoint_5": row[19],
                    "viewpoint_6": row[20]
                }

                # 帖子内容
                post_content = row[21] if row[21] else "No related post"

                # Add round title
                memory_content += f"#### Round {timestep}\n\n"

                # Add activation status
                if is_active:
                    memory_content += f"**Status**: Active\n\n"
                    memory_content += f"**Action**: {action}\n\n"
                    memory_content += f"**Reason**: {reason}\n\n"
                    memory_content += f"**Related Post**: {post_content[:100]}...\n\n" if len(post_content) > 100 else f"**Related Post**: {post_content}\n\n"
                else:
                    memory_content += f"**Status**: Inactive\n\n"

                # Add cognitive state
                memory_content += "**Cognitive State**:\n"
                memory_content += f"- mood: {mood_type} - {mood_value}\n"
                memory_content += f"- Emotion: {emotion_type} - {emotion_value}\n"
                memory_content += f"- Stance: {stance_type} - {stance_value}\n"
                memory_content += f"- thinking: {thinking_type} - {thinking_value}\n"
                memory_content += f"- Intention: {intention_type} - {intention_value}\n\n"

                # Add viewpoint support levels
                memory_content += "**Viewpoint Support Levels**:\n"
                for viewpoint_key, support_level in viewpoints.items():
                    if support_level and support_level != "Indifferent":
                        memory_content += f"- {viewpoint_key}: {support_level}\n"
                memory_content += "\n"

            return memory_content

        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 生成记忆内容时出错: {e}")
            import traceback
            agent_log.error(traceback.format_exc())
            # 确保返回一个有效的字符串
            return "No historical memory due to error."

    async def generate_memory_content_csv(self, current_timestep: int) -> str:
        """
        从CSV文件生成智能体的记忆内容，包括最近n轮的用户行为和认知状态

        Args:
            current_timestep: 当前时间步

        Returns:
            str: 记忆内容文本
        """

        if current_timestep == 1:
            return "No historical memory."
        try:


            # 计算开始轮次（最多获取前n轮）
            start_timestep = max(1, current_timestep - self.num_historical_memory)

            # 读取CSV文件
            df = pd.read_csv(self.csv_path)

            # 筛选当前用户的数据并按时间步排序
            user_df = df[(df['user_id'] == self.agent_id) & (df['timestep'] >= start_timestep) & (df['timestep'] < current_timestep)]
            user_df = user_df.sort_values(by='timestep', ascending=False)

            if current_timestep == 1 and user_df.empty:
                return "No historical memory."

            # 构建记忆内容
            memory_content = f"### Historical Memory for User {self.agent_id} (Steps {start_timestep} to {current_timestep-1})\n\n"

            # 处理每一行数据
            for _, row in user_df.iterrows():
                # 提取数据
                timestep = row['timestep']
                post_id = row['post_id']
                action = row['action']
                reason = row['reason']
                mood_type = row['mood_type']
                mood_value = row['mood_value']
                emotion_type = row['emotion_type']
                emotion_value = row['emotion_value']
                stance_type = row['stance_type']
                stance_value = row['stance_value']
                thinking_type = row['thinking_type']
                thinking_value = row['thinking_value']
                intention_type = row['intention_type']
                intention_value = row['intention_value']
                is_active = row['is_active']

                # 提取观点
                viewpoints = {
                    "viewpoint_1": row['viewpoint_1'],
                    "viewpoint_2": row['viewpoint_2'],
                    "viewpoint_3": row['viewpoint_3'],
                    "viewpoint_4": row['viewpoint_4'],
                    "viewpoint_5": row['viewpoint_5'],
                    "viewpoint_6": row['viewpoint_6']
                }

                # 帖子内容（CSV中可能没有这个字段，使用默认值）
                post_content = "No related post"

                # 添加轮次标题
                memory_content += f"#### Round {timestep}\n\n"

                # 添加激活状态
                if is_active:
                    memory_content += f"**Status**: Active\n\n"
                    memory_content += f"**Action**: {action}\n\n"
                    memory_content += f"**Reason**: {reason}\n\n"
                    memory_content += f"**Related Post ID**: {post_id}\n\n"
                else:
                    memory_content += f"**Status**: Inactive\n\n"

                # 添加认知状态
                memory_content += "**Cognitive State**:\n"
                memory_content += f"- mood: {mood_type} - {mood_value}\n"
                memory_content += f"- Emotion: {emotion_type} - {emotion_value}\n"
                memory_content += f"- Stance: {stance_type} - {stance_value}\n"
                memory_content += f"- Thinking: {thinking_type} - {thinking_value}\n"
                memory_content += f"- Intention: {intention_type} - {intention_value}\n\n"

                # 添加观点支持级别
                memory_content += "**Viewpoint Support Levels**:\n"
                for viewpoint_key, support_level in viewpoints.items():
                    if support_level and support_level != "Indifferent":
                        memory_content += f"- {viewpoint_key}: {support_level}\n"
                memory_content += "\n"

            return memory_content

        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 从CSV生成记忆内容时出错: {e}")
            import traceback
            agent_log.error(traceback.format_exc())
            # 确保返回一个有效的字符串
            return "No historical memory due to error."


    async def save_memory_data(self, timestep: int, is_active: bool):
        """
        保存智能体的记忆数据并更新认知知识图谱。
        同时生成并保存智能体的记忆内容，用于下一轮的决策。

        Args:
            timestep: 当前时间步
            is_active: 该智能体在当前时间步是否被激活
        """
        try:
            self.check_cognitive_profile()

            agent_log.info(f"Agent {self.agent_id} 第{self.step_counter}轮认知状态: {json.dumps(self.cognitive_profile, indent=2, ensure_ascii=False)}")

            # 确保非激活智能体有有效的current_action
            if not is_active:
                # 保存当前的current_action以备临时使用
                original_action = self.current_action.copy() if isinstance(self.current_action, dict) else {}
                original_cognitive_profile = self.cognitive_profile.copy()
                # 设置非激活状态的current_action
                self.current_action = {
                    "action_name": "inactive",
                    "arguments": {},
                    "reason": "inactive"
                }
                # 只保存用户信息，未激活用户不保存用户行为
                #await self.save_user_information()
                await self.save_user_action_dict(save_mode="db")

                # 恢复原始current_action，以避免丢失历史状态
                self.current_action = original_action
                self.cognitive_profile = original_cognitive_profile
            else:
                # 激活智能体保存用户信息和用户行为
                #await self.save_user_information()
                await self.save_user_action_dict(save_mode="db")

            # 保存完数据后，更新全局认知画像字典
            if hasattr(self, "users_cognitive_profile_dict"):
                self.users_cognitive_profile_dict[self.agent_id] = self.cognitive_profile.copy()
                agent_log.info(f"用户{self.agent_id}的认知画像已更新到全局字典")

            # 保存记忆内容到文件
            if not hasattr(self, "memory_dir"):
                self.memory_dir = os.path.join(os.getcwd(), "agent_memories")
                agent_log.info(f"设置记忆目录为: {self.memory_dir}")

            # 创建目录（如果不存在）
            os.makedirs(self.memory_dir, exist_ok=True)

            memory_file_path = os.path.join(self.memory_dir, f"agent_{self.agent_id}_memory_{timestep}.txt")
            # 确保 history_memory 不为 None
            memory_content = self.history_memory if self.history_memory is not None else "No historical memory."
            with open(memory_file_path, 'w', encoding='utf-8') as f:
                f.write(memory_content)
            agent_log.info(f"已保存用户{self.agent_id}在时间步{timestep}的记忆内容到: {memory_file_path}")

        except Exception as e:
            agent_log.error(f"保存用户数据时出错: {e}")
            import traceback
            agent_log.error(traceback.format_exc())


    async def perform_action_by_llm(self, is_individual: bool = False, is_target: bool = False, save_mode: str = "db", prompt_mode: str = "asce"):
        """使用LLM执行动作"""

        if not hasattr(self, 'env'):
            agent_log.error("错误：环境未初始化")
            return ""

        try:
            if is_individual and not is_target:
                env_prompt = self.individual_context
                openai_messages = await self._build_user_message(env_prompt, save_mode, prompt_mode)

                # 保存最后一次的提示词，用于重试
                self.last_prompt = openai_messages

            elif is_individual and is_target:
                env_prompt = self.individual_context
                target_content = self.target_content
                openai_messages = await self._build_user_message_label(env_prompt, target_content)
                # print (f"openai_messages:{openai_messages}")
                # import pdb;pdb.set_trace()

                # 保存最后一次的提示词，用于重试
                self.last_prompt = openai_messages
            else:
                env_prompt = await asyncio.wait_for(self.env.to_text_prompt(), timeout=10)
                openai_messages = await self._build_user_message(env_prompt, save_mode, prompt_mode)
                #print (f"openai_messages:{openai_messages}")

                # 保存最后一次的提示词，用于重试
                self.last_prompt = openai_messages

            # 步骤3: 验证和格式化消息
            try:
                # 验证消息格式
                openai_messages = await self._validate_messages(openai_messages)
                if not openai_messages:
                    return ""
                # 对于DeepSeek模型，应用特定的格式处理
                if self.is_deepseek_model:
                    openai_messages = await self._prepare_deepseek_messages(openai_messages)
            except Exception as e:
                return ""

            # 步骤4: 调用模型
            response = None
            try:
                # 根据模型类型调用不同的API
                if self.is_local_model:
                    # 如果是本地模型，使用专用的API调用函数
                    response = await self._call_local_model_api(openai_messages)
                else:
                    # 否则使用标准的模型后端
                    response = self.model_backend.run(openai_messages)
            except asyncio.TimeoutError:
                #print(f"步骤4: 模型调用超时 - agent_id={self.agent_id}")
                return ""
            except Exception as e:
                #print(f"步骤4: 模型调用失败 - agent_id={self.agent_id}, 错误: {str(e)}")
                import traceback
                #print(f"步骤4: 错误详情: {traceback.format_exc()}")
                return ""

            action_list = []
            #print(f"[DEBUG]RAW Response: {response}")

            # 安全解析 assistant_content
            try:
                # 如果 response 是包含 'choices' 字段的字典（例如标准OpenAI返回格式）
                # print(f"response:{response}")
                # import pdb;pdb.set_trace()
                if isinstance(response, dict) and "choices" in response:
                    assistant_content = response['choices'][0]['message']['content']
                # 如果 response 是字符串形式，直接使用
                elif isinstance(response, str):
                    assistant_content = response
                # 如果 response 是字典，但没有 'choices'，尝试读取 'content' 字段
                elif isinstance(response, dict) and "content" in response:
                    assistant_content = response['content']
                else:
                    # 其他未知格式时记录警告并使用空字符串
                    agent_log.warning(f"无法解析 response 内容，类型: {type(response)}，内容: {response}")
                    assistant_content = ""

                self.raw_response = assistant_content
                agent_log.debug(f"【DEBUG】解析前 assistant_content: {assistant_content}")

            except Exception as e:
                agent_log.error(f"解析 assistant_content 出错: {str(e)}")
                assistant_content = ""
                self.raw_response = ""

            #assistant_content = response['choices'][0]['message']['content']
            self.raw_response = assistant_content
            agent_log.debug(f"【DEBUG】解析前assistant_content: {assistant_content}")
            # print(f"self.is_local_model: {self.is_local_model}")
            try:
                if self.is_openai_model:
                    action_list = await self._process_model_response(assistant_content, "openai")
                elif self.is_deepseek_model:
                    action_list = await self._process_model_response(assistant_content, "deepseek")
                elif self.is_local_model:
                    action_list = await self._process_model_response(assistant_content, "vllm")

                    # 执行解析出的动作（针对vLLM模型）
                    if action_list and isinstance(action_list, list) and len(action_list) > 0:
                        for action in action_list:
                            if isinstance(action, dict) and "name" in action and "arguments" in action:
                                action_name = action["name"]
                                arguments = action["arguments"]

                                # 跳过unknown_action
                                if action_name == "unknown_action":
                                    agent_log.warning(f"跳过执行未知动作: {action_name}")
                                    continue

                                # 查找匹配的函数
                                matching_action = None
                                for func_name in dir(self.env.action):
                                    if func_name.startswith('_'):
                                        continue

                                    if func_name == action_name:
                                        matching_action = func_name
                                        break

                                if matching_action:
                                    # 如果是关注操作，并且没有指定step_number，则添加当前轮次
                                    if matching_action == "follow" and "step_number" not in arguments:
                                        try:
                                            step_number = int(os.environ.get("TIME_STAMP", 0))
                                            arguments["step_number"] = step_number
                                        except (ValueError, TypeError):
                                            pass

                                    # 执行动作
                                    agent_log.info(f"执行vLLM解析出的动作: {matching_action}，参数: {arguments}")
                                    try:
                                        result = await getattr(self.env.action, matching_action)(**arguments)
                                        agent_log.info(f"动作执行结果: {result}")

                                        # 更新agent_graph
                                        if hasattr(self, 'agent_graph') and self.agent_graph is not None:
                                            self.perform_agent_graph_action(matching_action, arguments)
                                    except Exception as e:
                                        agent_log.error(f"执行动作失败: {matching_action}，错误: {str(e)}")
                                else:
                                    agent_log.warning(f"找不到匹配的函数: {action_name}")
            except Exception as e:
                agent_log.error(f"响应处理失败 - agent_id={self.agent_id}, 错误: {str(e)}")
            agent_log.info(f"第{self.step_counter}轮响应解析后动作：\n{self.current_action}")



        except Exception as e:
            agent_log.error(f"响应处理失败 - agent_id={self.agent_id}, 错误: {str(e)}")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        """根据动作类型修改代理图结构：
        如果动作是取消关注(unfollow)，则从图中删除边；
        如果动作是关注(follow)，则在图中添加边。
        """
        # 获取被关注者ID
        followee_id = arguments.get("followee_id")
        if followee_id is None:
            return

        # 根据动作类型执行相应操作
        if "unfollow" in action_name:
            self.agent_graph.remove_edge(self.agent_id, followee_id)
            agent_log.debug(f"Agent {self.agent_id} unfollowed {followee_id}")
        elif "follow" in action_name:
            self.agent_graph.add_edge(self.agent_id, followee_id)
            agent_log.debug(f"Agent {self.agent_id} followed {followee_id}")


    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.agent_id}, "
                f"model_type={self.model_type.value})")

    async def _extract_functions(self, content):
        """从内容中提取函数调用信息"""
        try:
            # 检查是否包含```json```格式的JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

            if json_match:
                # 提取并解析JSON字符串
                try:
                    parsed_json = json.loads(json_match.group(1))
                    if "functions" in parsed_json and parsed_json["functions"]:
                        return parsed_json["functions"][0]
                except json.JSONDecodeError:
                    agent_log.warning("无法解析JSON代码块")

            # 尝试直接解析content为JSON
            try:
                parsed_json = json.loads(content)
                if "functions" in parsed_json and parsed_json["functions"]:
                    return parsed_json["functions"][0]
            except json.JSONDecodeError:
                # 使用正则表达式查找functions字段
                functions_match = re.search(r'"functions"\s*:\s*\[\s*{\s*"name"\s*:\s*"([^"]+)"', content)
                if functions_match:
                    return {"name": functions_match.group(1), "arguments": {}}

        except Exception as e:
            agent_log.error(f"提取函数调用信息错误: {e}")

        return None

    async def _process_function_call(self, function_call):
        """
        处理函数调用

        参数:
            function_call (dict): 函数调用信息

        返回:
            dict: 函数调用结果
        """
        try:
            function_name = function_call.get("name", "")
            arguments = function_call.get("arguments", {})

            # 更新当前动作记录
            self.current_action = {
                "action_name": function_name,
                "arguments": arguments,
                "reason": self.reason
            }

            # 提取认知状态

            # 匹配真实动作
            matching_action = None
            for func_name in dir(self.env.action):
                if func_name.startswith('_'):
                    continue

                func = getattr(self.env.action, func_name)

                if callable(func) and not func_name.startswith('_'):
                    if func_name == function_name:
                        matching_action = func_name
                        break

            if matching_action:
                #print(f"==DEBUG== 调用{matching_action}方法")

                # 如果是关注操作，并且没有指定step_number，则添加当前轮次
                if matching_action == "follow" and "step_number" not in arguments:
                    # 尝试从环境变量获取时间步
                    try:
                        step_number = int(os.environ.get("TIME_STAMP", 0))
                        arguments["step_number"] = step_number
                    except (ValueError, TypeError):
                        pass

                result = await getattr(self.env.action, matching_action)(**arguments)

                # 记录操作到agent_graph（如果有）
                if hasattr(self, 'agent_graph') and self.agent_graph is not None:
                    self.perform_agent_graph_action(matching_action, arguments)
                return result
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}

        except Exception as e:
            agent_log.error(f"处理函数调用出错: {str(e)}")
            import traceback
            agent_log.error(f"错误详情: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def _process_model_response(self, response, model_type="default", is_retry=False, retry_count=0, max_retries=None):
        """
        统一处理不同模型的响应

        参数:
            response: 模型的响应
            model_type: 模型类型，可以是"openai"、"deepseek"或"vllm"
            is_retry: 是否是重试调用
            retry_count: 当前重试次数
            max_retries: 最大重试次数，如果为None则使用默认值

        返回:
            处理后的内容
        """
        # 如果没有指定最大重试次数，使用默认值
        if max_retries is None:
            max_retries = getattr(self, 'llm_retry_count', 2)

        # 记录响应尝试
        if not is_retry and hasattr(self, 'response_quality') and self.response_quality.enabled:
            self.response_quality.record_response(self.agent_id)
            self.response_total_count += 1

        try:
            agent_log.debug(f"RAW content: {str(response)}")
            action_list = []

            # 根据模型类型调用相应的处理函数


            if "</think>" in response:
                think_content, response_content = response.split("</think>", 1)
                response_str = response_content.strip()
                formatted_content = f"Agent {self.agent_id} ({self.user_info.name}) thinking content for round {self.step_counter}:{think_content.strip()}"
                self.think_content = formatted_content

            else:
                response_str = response.strip()
                agent_log.warning(f"响应内容中未找到</think>分隔符，未提取思考内容")

            import copy
            cognitive_profile_backup = copy.deepcopy(self.cognitive_profile)

            # 尝试解析响应
            action_list = await self._parse_vllm_response(response_str)

            # 检查解析是否成功
            parse_success = self._validate_parse_result(action_list, self.cognitive_profile, cognitive_profile_backup)
            if parse_success:
                agent_log.info(f"智能体{self.agent_id}在第{self.step_counter}轮的认知状态，解析成功\n{self.cognitive_profile}")

                # 确保返回的action_list是一个列表，包含当前动作
                if hasattr(self, 'current_action') and self.current_action:
                    # 如果current_action已经设置，将其添加到action_list中
                    if not isinstance(action_list, list):
                        action_list = []

                    # 确保action_list中包含当前动作
                    action_found = False
                    for action in action_list:
                        if isinstance(action, dict) and action.get('name') == self.current_action.get('action_name'):
                            action_found = True
                            break

                        if not action_found:
                            # 将current_action转换为action_list格式
                            action_list.append({
                            'name': self.current_action.get('action_name', 'unknown_action'),
                            'arguments': self.current_action.get('arguments', {})
                            })
                        agent_log.info(f"将当前动作添加到action_list: {self.current_action.get('action_name')}")
            else:
                agent_log.warning(f"智能体{self.agent_id}在第{self.step_counter}轮的认知状态，解析失败\n{self.cognitive_profile}")
                
                return action_list

        except Exception as e:
            import traceback
            agent_log.error(f"处理模型响应失败 - agent_id={self.agent_id}, 错误: {str(e)}")
            agent_log.error(f"错误详情: {traceback.format_exc()}")
            return [{
                "name": "unknown_action",
                "arguments": {}
            }]


    async def _parse_vllm_response(self, response):
        """
        处理VLLM本地模型的响应 - 优化版本

        使用ResponseParser类来提高解析的稳定性和准确性，避免多种解析策略之间的冲突。
        支持解析失败时的重试机制和解析统计。

        参数:
            response: VLLM模型的响应

        返回:
            处理后的函数调用列表
        """
        import copy
        # 创建当前认知状态的备份
        cognitive_profile_backup = copy.deepcopy(self.cognitive_profile)
        agent_log.info(f"[COG_DEBUG] 前一轮的认知: {json.dumps(cognitive_profile_backup, indent=2)}")

        # 初始化响应解析器
        parser = ResponseParser(
            normalization_map=NORMALIZATION_MAP,
            default_cognitive_profile=self.DEFAULT_COGNITIVE_PROFILE,
            normalize_support_level_func=self._normalize_support_level,
            cognition_space_dict=self.cognition_space_dict
        )

        # 提取内容
        content = self._extract_content_from_response(response)
        if not content:
            agent_log.error(f"无法从响应中提取内容")
            if self.collect_parse_stats:
                self.parse_stats.record_failure(self.agent_id)
                self.parse_failure_count += 1
            return [{
                "name": "unknown_action",
                "arguments": {}
            }]

        # 记录原始响应内容，用于重试
        original_content = content

        # 解析重试循环
        retry_count = 0
        max_retries = getattr(self, 'parse_retries', 2)
        success = False

        while retry_count <= max_retries:
            try:
                # 如果是重试，记录重试次数
                if retry_count > 0:
                    agent_log.info(f"开始第 {retry_count} 次解析重试")
                    if self.collect_parse_stats:
                        self.parse_stats.record_retry(self.agent_id)
                        self.parse_retry_count += 1

                # 步骤1: 从响应中提取JSON数据
                json_data, strategy = parser.extract_json_from_response(content)
                agent_log.info(f"使用策略 '{strategy}' 提取JSON数据")

                if self.collect_parse_stats:
                    self.parse_stats.record_attempt(self.agent_id, strategy)

                # 步骤2: 提取理由
                reason = parser.extract_reason_from_response(content, json_data)
                self.reason = reason
                agent_log.debug(f"提取的理由: {reason}")

                # 步骤3: 直接处理opinion数组（如果存在）
                opinion_processed = False
                if "opinion" in json_data and isinstance(json_data["opinion"], list):
                    agent_log.info(f"发现opinion数组，开始处理")
                    # 确保认知状态中存在opinion字段
                    if "opinion" not in self.cognitive_profile:
                        self.cognitive_profile["opinion"] = {}

                    # 遍历opinion数组
                    for item in json_data["opinion"]:
                        if not isinstance(item, dict):
                            continue

                        # 处理包含viewpoint_n和type_support_levels的格式
                        viewpoint_keys = [key for key in item if key.startswith("viewpoint_")]
                        if viewpoint_keys and "type_support_levels" in item:
                            for key in viewpoint_keys:
                                support_level = item.get("type_support_levels")
                                if isinstance(support_level, str):
                                    # 规范化支持级别
                                    normalized_support = self._normalize_support_level(support_level)
                                    self.cognitive_profile["opinion"][key] = normalized_support
                                    agent_log.info(f"从数组中提取观点: {key} = {normalized_support}")
                                    opinion_processed = True

                # 步骤4: 处理认知状态
                updated_cognitive_profile = parser.process_cognitive_state(json_data, self.cognitive_profile)

                # 步骤5: 如果提取到有效的认知状态，更新当前认知状态
                cognitive_state_valid = False
                if updated_cognitive_profile:
                    # 检查更新后的认知状态是否有效
                    is_valid = self._validate_cognitive_profile(updated_cognitive_profile)
                    if is_valid:
                        self.cognitive_profile = updated_cognitive_profile
                        agent_log.info(f"成功更新认知状态")
                        agent_log.info(f"智能体{self.agent_id}第{self.step_counter}轮的认知状态: {self.cognitive_profile}")
                        cognitive_state_valid = True
                    else:
                        agent_log.warning(f"更新后的认知状态无效，保留原有状态")

                # 步骤6: 提取函数调用
                functions = parser.extract_functions_from_response(content, json_data)

                # 检查解析是否成功
                # 成功条件: 有效的认知状态或成功处理了opinion数组，以及有效的函数调用
                if (cognitive_state_valid or opinion_processed) and functions:
                    agent_log.info(f"解析成功，提取到 {len(functions)} 个函数调用")
                    success = True

                    if self.collect_parse_stats:
                        self.parse_stats.record_success(self.agent_id, strategy)
                        self.parse_success_count += 1

                    # 更新当前动作
                    if functions[0]:
                        self.current_action = {
                            "action_name": functions[0].get("name", "unknown_action"),
                            "arguments": functions[0].get("arguments", {}),
                            "reason": self.reason
                        }
                    return functions
                else:
                    # 如果解析失败但还有重试机会，继续重试
                    if retry_count < max_retries:
                        agent_log.warning(f"解析不完整，准备重试 ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        # 恢复备份的认知状态
                        self.cognitive_profile = copy.deepcopy(cognitive_profile_backup)
                        # 使用原始内容重试
                        content = original_content
                        continue
                    else:
                        # 如果没有重试机会了，但至少有函数调用，返回函数调用
                        if functions:
                            agent_log.warning(f"认知状态解析失败，但有函数调用，返回函数调用")

                            if self.collect_parse_stats:
                                self.parse_stats.record_failure(self.agent_id)
                                self.parse_failure_count += 1

                            # 更新当前动作
                            if functions[0]:
                                self.current_action = {
                                    "action_name": functions[0].get("name", "unknown_action"),
                                    "arguments": functions[0].get("arguments", {}),
                                    "reason": self.reason
                                }
                            return functions
                        else:
                            # 如果没有函数调用，返回默认动作
                            agent_log.warning(f"解析完全失败，返回默认动作")

                            if self.collect_parse_stats:
                                self.parse_stats.record_failure(self.agent_id)
                                self.parse_failure_count += 1

                            # 恢复备份的认知状态
                            self.cognitive_profile = cognitive_profile_backup
                            return [{
                                "name": "unknown_action",
                                "arguments": {}
                            }]

            except Exception as e:
                agent_log.error(f"VLLM响应处理失败 - 重试{retry_count}/{max_retries} - agent_id={self.agent_id} error={str(e)}")
                import traceback
                agent_log.error(traceback.format_exc())

                # 如果还有重试机会，继续重试
                if retry_count < max_retries:
                    retry_count += 1
                    # 恢复备份的认知状态
                    self.cognitive_profile = copy.deepcopy(cognitive_profile_backup)
                    # 使用原始内容重试
                    content = original_content
                    continue
                else:
                    # 所有重试都失败
                    if self.collect_parse_stats:
                        self.parse_stats.record_failure(self.agent_id)
                        self.parse_failure_count += 1

                    # 恢复备份的认知状态
                    self.cognitive_profile = cognitive_profile_backup
                    return [{
                        "name": "unknown_action",
                        "arguments": {}
                    }]

        # 如果执行到这里，说明所有重试都失败
        agent_log.error(f"所有解析重试都失败，返回默认动作")

        if self.collect_parse_stats:
            self.parse_stats.record_failure(self.agent_id)
            self.parse_failure_count += 1

        # 恢复备份的认知状态
        self.cognitive_profile = cognitive_profile_backup
        return [{
            "name": "unknown_action",
            "arguments": {}
        }]

    def _extract_content_from_response(self, response):
        """
        从响应中提取内容

        参数:
            response: 模型响应

        返回:
            提取的内容字符串
        """
        try:
            # 检查响应是否为空或包含错误
            if response is None:
                agent_log.warning(f"响应为None，无法提取内容")
                return ""

            if isinstance(response, dict) and 'error' in response:
                error_msg = response.get('error', '')
                agent_log.warning(f"响应包含错误: {error_msg}")
                # 如果错误消息为空，返回空字符串
                if not error_msg:
                    return ""
                # 否则返回错误消息，可能包含有用信息
                return f"Error: {error_msg}"

            # 如果是对象并有choices属性
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content

            # 如果是字典并有choices键
            elif isinstance(response, dict) and 'choices' in response:
                if isinstance(response['choices'], list) and len(response['choices']) > 0:
                    if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                        content = response['choices'][0]['message']['content']
                        # 检查内容是否为空
                        if not content or content.strip() == "":
                            agent_log.warning(f"响应内容为空")
                            return ""
                        return content
                    elif 'text' in response['choices'][0]:
                        return response['choices'][0]['text']
                else:
                    agent_log.warning(f"响应中的choices列表为空")
                    return ""

            # 如果是字符串
            elif isinstance(response, str):
                # 检查字符串是否为空
                if not response or response.strip() == "":
                    agent_log.warning(f"响应字符串为空")
                    return ""
                return response

            # 其他情况，转换为字符串
            result = str(response)
            if result == "{}":
                agent_log.warning(f"响应为空对象")
                return ""
            return result

        except Exception as e:
            agent_log.error(f"提取内容时出错: {e}")
            import traceback
            agent_log.error(traceback.format_exc())
            return ""

    def _validate_parse_result(self, action_list, current_profile, backup_profile):
        """
        验证解析结果是否有效

        参数:
            action_list: 解析出的动作列表
            current_profile: 当前认知状态
            backup_profile: 备份的认知状态

        返回:
            布尔值，表示解析是否成功
        """
        # 检查动作列表是否有效
        if not action_list or not isinstance(action_list, list) or len(action_list) == 0:
            agent_log.warning(f"动作列表无效: {action_list}")
            return False

        # 检查第一个动作是否有效
        first_action = action_list[0]
        if not isinstance(first_action, dict) or "name" not in first_action:
            agent_log.warning(f"第一个动作无效: {first_action}")
            return False

        # 检查动作名称是否有效
        if first_action["name"] == "unknown_action":
            agent_log.warning(f"动作名称无效: {first_action['name']}")
            return False

        # 检查认知状态的基本结构
        if not self._validate_cognitive_profile_structure(current_profile):
            agent_log.warning(f"认知状态结构无效")
            return False

        # 检查认知状态中的opinion字段
        if "opinion" in current_profile:
            # 检查opinion是否为字典类型
            if not isinstance(current_profile["opinion"], dict):
                agent_log.warning(f"opinion不是字典类型: {type(current_profile['opinion'])}")
                return False

            # 检查是否有多于6个观点
            viewpoint_keys = [k for k in current_profile["opinion"].keys() if k.startswith("viewpoint_")]
            if len(viewpoint_keys) > 6:
                agent_log.warning(f"观点数量超过6个: {viewpoint_keys}")
                # 删除多余的观点
                for key in viewpoint_keys:
                    if int(key.split('_')[1]) > 6:
                        agent_log.warning(f"删除多余观点: {key}")
                        del current_profile["opinion"][key]

            # 检查是否缺少观点
            for i in range(1, 7):
                key = f"viewpoint_{i}"
                if key not in current_profile["opinion"]:
                    agent_log.warning(f"缺少观点: {key}")
                    # 使用备份中的观点或默认值
                    if backup_profile and "opinion" in backup_profile and key in backup_profile["opinion"]:
                        current_profile["opinion"][key] = backup_profile["opinion"][key]
                    else:
                        current_profile["opinion"][key] = "fail"

            # 检查观点支持级别是否有效
            if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict and "opinion_support_levels" in self.cognition_space_dict:
                valid_support_levels = self.cognition_space_dict["opinion_support_levels"]
                for i in range(1, 7):
                    key = f"viewpoint_{i}"
                    if key in current_profile["opinion"] and current_profile["opinion"][key] not in valid_support_levels:
                        agent_log.warning(f"观点{key}的支持级别无效: {current_profile['opinion'][key]}")
                        # 尝试规范化支持级别
                        normalized_level = self._normalize_support_level(current_profile["opinion"][key])
                        if normalized_level in valid_support_levels:
                            current_profile["opinion"][key] = normalized_level
                        else:
                            # 使用备份中的支持级别或默认值
                            if backup_profile and "opinion" in backup_profile and key in backup_profile["opinion"]:
                                current_profile["opinion"][key] = backup_profile["opinion"][key]
                            else:
                                current_profile["opinion"][key] = "Indifferent"  # 默认为中立

        # 检查认知状态的其他维度
        for field in ["mood", "emotion", "stance", "thinking", "intention"]:
            if field in current_profile:
                # 检查维度是否为字典类型
                if not isinstance(current_profile[field], dict):
                    agent_log.warning(f"{field}不是字典类型: {type(current_profile[field])}")
                    # 尝试修复结构
                    if isinstance(current_profile[field], str):
                        # 如果是字符串，将其转换为字典
                        value = current_profile[field]
                        current_profile[field] = {"type": self._normalize_cognitive_type(field, value), "value": value}
                    else:
                        # 使用备份中的值或默认值
                        if backup_profile and field in backup_profile and isinstance(backup_profile[field], dict):
                            current_profile[field] = backup_profile[field]
                        else:
                            current_profile[field] = {"type": "fail", "value": "fail"}

                # 检查type和value是否存在
                if "type" not in current_profile[field] or "value" not in current_profile[field]:
                    agent_log.warning(f"{field}缺少type或value字段")
                    # 使用备份中的值或默认值
                    if backup_profile and field in backup_profile and isinstance(backup_profile[field], dict):
                        current_profile[field] = backup_profile[field]
                    else:
                        current_profile[field] = {"type": "fail", "value": "fail"}

                # 检查type和value是否有效
                if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict:
                    # 检查type是否有效
                    if f"{field}" in self.cognition_space_dict and "type_list" in self.cognition_space_dict[field]:
                        valid_types = self.cognition_space_dict[field]["type_list"]
                        if current_profile[field]["type"] not in valid_types:
                            agent_log.warning(f"{field}的type无效: {current_profile[field]['type']}")
                            # 尝试规范化type
                            normalized_type = self._normalize_cognitive_type(field, current_profile[field]["type"])
                            if normalized_type in valid_types:
                                current_profile[field]["type"] = normalized_type
                            else:
                                # 使用备份中的type或默认值
                                if backup_profile and field in backup_profile and "type" in backup_profile[field]:
                                    current_profile[field]["type"] = backup_profile[field]["type"]
                                else:
                                    current_profile[field]["type"] = valid_types[0]  # 使用第一个有效类型

                    # 检查value是否有效
                    if f"{field}" in self.cognition_space_dict and "value_list" in self.cognition_space_dict[field]:
                        valid_values = self.cognition_space_dict[field]["value_list"]
                        if current_profile[field]["value"] not in valid_values:
                            agent_log.warning(f"{field}的value无效: {current_profile[field]['value']}")
                            # 尝试规范化value
                            # 如果type和value相同，尝试从值域中选择一个合适的值
                            if current_profile[field]["type"] == current_profile[field]["value"]:
                                # 尝试从当前type对应的值域中选择一个值
                                type_value = current_profile[field]["type"]
                                if type_value in valid_types:
                                    # 获取该类型对应的值域
                                    type_values = []
                                    for v in valid_values:
                                        if v.lower().startswith(type_value.lower()):
                                            type_values.append(v)
                                    if type_values:
                                        current_profile[field]["value"] = type_values[0]
                                    else:
                                        # 如果没有匹配的值，使用备份中的value或默认值
                                        if backup_profile and field in backup_profile and "value" in backup_profile[field]:
                                            current_profile[field]["value"] = backup_profile[field]["value"]
                                        else:
                                            current_profile[field]["value"] = valid_values[0]  # 使用第一个有效值
                            else:
                                # 使用备份中的value或默认值
                                if backup_profile and field in backup_profile and "value" in backup_profile[field]:
                                    current_profile[field]["value"] = backup_profile[field]["value"]
                                else:
                                    current_profile[field]["value"] = valid_values[0]  # 使用第一个有效值

        # 如果认知状态没有变化，但动作有效，仍然认为解析成功
        return True

    def _validate_cognitive_profile_structure(self, profile):
        """
        验证认知状态的基本结构

        参数:
            profile: 要验证的认知状态

        返回:
            布尔值，表示结构是否有效
        """
        # 检查是否为字典类型
        if not isinstance(profile, dict):
            agent_log.warning(f"认知状态不是字典类型: {type(profile)}")
            return False

        # 检查必要字段是否存在
        required_fields = ["mood", "emotion", "stance", "thinking", "intention", "opinion"]
        for field in required_fields:
            if field not in profile:
                agent_log.warning(f"认知状态缺少必要字段: {field}")
                return False

        return True

    async def _call_llm_with_last_prompt(self):
        """
        使用上一次的提示词重新调用LLM

        返回:
            新的LLM响应
        """
        try:
            # 检查是否有上一次的提示词
            if not hasattr(self, 'last_prompt') or not self.last_prompt:
                agent_log.warning(f"没有上一次的提示词，无法重新调用LLM")
                return None

            # 重新调用LLM
            agent_log.info(f"使用上一次的提示词重新调用LLM")

            # 根据模型类型选择不同的调用方式
            if self.is_openai_model:
                # 使用OpenAI模型
                response = await self.model_backend.generate(self.last_prompt)
                return response
            elif self.is_deepseek_model:
                # 使用DeepSeek模型
                response = await self.model_backend.generate(self.last_prompt)
                return response
            elif self.is_local_model and self.multi_api_handler:
                # 使用本地模型和多端API处理器
                response = await self.multi_api_handler.call_model_with_retry(self.last_prompt, self.agent_id)
                return response
            else:
                agent_log.warning(f"不支持的模型类型，无法重新调用LLM")
                return None

        except Exception as e:
            agent_log.error(f"重新调用LLM失败: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return None

    def _validate_cognitive_profile(self, profile):
        """
        验证认知状态是否有效

        参数:
            profile: 要验证的认知状态

        返回:
            布尔值，表示认知状态是否有效
        """
        # 检查基本结构
        if not isinstance(profile, dict):
            return False

        # 检查必要字段
        required_fields = ["mood", "emotion", "stance", "thinking", "intention", "opinion"]
        if not all(field in profile for field in required_fields):
            return False

        # 检查基本认知字段是否有效
        for field in ["mood", "emotion", "stance", "thinking", "intention"]:
            if not isinstance(profile[field], dict) or "type" not in profile[field] or "value" not in profile[field]:
                return False
            # 检查值是否为“fail”
            if profile[field]["type"] == "fail" or profile[field]["value"] == "fail":
                return False

        # 检查opinion字段
        if not isinstance(profile["opinion"], dict):
            return False

        # 检查是否有所有的viewpoint键
        for i in range(1, 7):
            key = f"viewpoint_{i}"
            if key not in profile["opinion"] or profile["opinion"][key] == "fail":
                return False

        return True

    def _normalize_support_level(self, support_level):
        """
        规范化支持级别

        参数:
            support_level: 原始支持级别

        返回:
            规范化后的支持级别
        """
        if not support_level:
            return "Indifferent"

        # 将支持级别转换为小写
        support_level = str(support_level).lower()

        # 定义支持级别映射
        support_map = {
            "strongly support": "Strongly Support",
            "moderate support": "Moderate Support",
            "indifferent": "Indifferent",
            "do not support": "Do Not Support",
            "strongly opposition": "Strongly Opposition"
        }

        # 匹配支持级别
        for key, value in support_map.items():
            if key in support_level:
                return value

        # 如果没有匹配到，尝试使用认知空间中的支持级别
        if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict and "opinion_support_levels" in self.cognition_space_dict:
            valid_support_levels = self.cognition_space_dict["opinion_support_levels"]

            # 尝试模糊匹配
            for valid_level in valid_support_levels:
                if valid_level.lower() in support_level or support_level in valid_level.lower():
                    return valid_level

        # 如果还是没有匹配到，返回默认值
        return "Indifferent"

    def _normalize_cognitive_type(self, field, value):
        """
        规范化认知类型

        参数:
            field: 认知维度字段名称
            value: 原始值

        返回:
            规范化后的类型
        """
        if not value:
            return "fail"

        # 将值转换为小写
        value = str(value).lower()

        # 如果有认知空间字典，尝试使用其中的类型
        if hasattr(self, 'cognition_space_dict') and self.cognition_space_dict and field in self.cognition_space_dict and "type_list" in self.cognition_space_dict[field]:
            valid_types = self.cognition_space_dict[field]["type_list"]

            # 精确匹配
            for valid_type in valid_types:
                if valid_type.lower() == value:
                    return valid_type

            # 模糊匹配
            for valid_type in valid_types:
                if valid_type.lower() in value or value in valid_type.lower():
                    return valid_type

            # 如果没有匹配到，返回第一个有效类型
            if valid_types:
                return valid_types[0]

        # 如果没有认知空间字典或没有匹配到，返回原始值
        return value

    async def on_timestep_end(self, timestep):
        """
        在每个时间步结束时调用的方法

        参数:
            timestep: 当前时间步
        """
        # 更新step_counter以跟踪当前轮次
        self.step_counter = timestep
        # 添加日志输出当前认知状态
        agent_log.debug(f"[TIMESTEP_END] 代理{self.agent_id}更新轮次为{timestep}")
        agent_log.debug(f"[TIMESTEP_END] 当前认知状态: {json.dumps(self.cognitive_profile, indent=2)}")



    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        """
        通过提供的数据执行代理动作的方法。
        参数:
            func_name: 要执行的函数名称
            *args: 可变位置参数
            **kwargs: 可变关键字参数
        返回:
            函数执行的结果
        """
        # 获取函数列表并查找匹配函数
        function_list = self.env.action.get_openai_function_list()
        for func_item in function_list:
            if func_item.func.__name__ == func_name:
                # 执行函数并返回结果
                result = await func_item.func(*args, **kwargs)
                agent_log.debug(f"Agent {self.agent_id}: {result}")
                return result

        # 未找到匹配函数时抛出异常
        raise ValueError(f"Function {func_name} not found in the list.")

    async def _call_local_model_api(self, messages):
        """
        调用本地模型API

        参数:
            messages: 消息列表

        返回:
            模型响应
        """
        # 如果有多API处理器，优先使用它
        if hasattr(self, 'multi_api_handler') and self.multi_api_handler is not None:
            try:
                agent_log.debug(f"代理{self.agent_id}: 使用MultiApiHandler调用模型")
                return await self.multi_api_handler.call_model_with_retry(messages, self.agent_id)
            except Exception as e:
                agent_log.error(f"代理{self.agent_id}: 使用MultiApiHandler调用模型失败: {str(e)}")
                agent_log.error(f"将尝试使用常规方法调用API")

        # 原有的API调用逻辑
        try:
            # 确保API基础URL正确
            api_base = self.local_model_api_base
            api_base = api_base.rstrip('/')

            # 构建请求URL
            url = f"{api_base}/chat/completions"

            # 构建请求头
            headers = {
                "Content-Type": "application/json"
            }

            # 构建请求数据
            #print(f"messages: {messages}")
            data = {
                "model": self.model_type,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens, 
            }

            # 发送请求
            agent_log.debug(f"发送请求到本地模型API: {url}")
            agent_log.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)[:200]}")

            # 使用aiohttp发送异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    #print(f"response_json: {response_json}")
                    agent_log.debug(f"本地模型API响应: {json.dumps(response_json, ensure_ascii=False)[:200]}")
                    return response_json

        except Exception as e:
            agent_log.error(f"调用本地模型API失败: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return {"error": str(e)}

    async def init_save_user_information(self):
        """
        保存用户信息到user_information表，仅在用户初始化时执行一次

        从用户信息对象中提取详细信息并保存到数据库中
        """
        try:


            # 从用户信息中提取数据
            user_id = self.agent_id  # 使用agent_id作为user_id
            persona = self.user_info.profile.get("persona", "")

            # 从other_info中提取其他字段
            other_info = self.user_info.profile.get("other_info", {})
            age = other_info.get("age", "")
            gender = other_info.get("gender", "")
            mbti = other_info.get("mbti", "")
            country = other_info.get("country", "")
            profession = other_info.get("profession", "")

            # 处理interested_topics，确保它是字符串
            interested_topics = other_info.get("interested_topics", [])
            if isinstance(interested_topics, list):
                interested_topics = json.dumps(interested_topics)

            # 调用平台方法保存用户信息
            result = await self.env.platform.save_user_information(
                user_id,
                persona,
                age,
                gender,
                mbti,
                country,
                profession,
                str(interested_topics)
            )

            if result.get("success"):
                agent_log.debug(f"成功保存用户信息: {user_id}")
                return True
            else:
                agent_log.warning(f"保存用户信息失败: {result.get('error', '未知错误')}")
                return False

        except Exception as e:
            agent_log.error(f"保存用户信息时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return False

    async def initial_save_user_action(self,timestep):

        try:


            # 获取用户ID
            user_id = self.agent_id


            action = "initial"
            reason = "initial"
            action_info = 'initial'
            post_id = 0
            agent_log.debug(f"初始化智能体 {user_id}")


            # 从认知状态中提取各项值
            # 情感
            mood = self.cognitive_profile.get("mood", {})
            mood_type = mood.get("type", "")
            mood_value = mood.get("value", "")

            # 情绪
            emotion = self.cognitive_profile.get("emotion", {})
            emotion_type = emotion.get("type", "")
            emotion_value = emotion.get("value", "")

            # 立场
            stance = self.cognitive_profile.get("stance", {})
            stance_type = stance.get("type", "")
            stance_value = stance.get("value", "")

            # 认知
            thinking = self.cognitive_profile.get("thinking", {})
            thinking_type = thinking.get("type", "")
            thinking_value = thinking.get("value", "")

            # 意图
            intention = self.cognitive_profile.get("intention", {})
            intention_type = intention.get("type", "")
            intention_value = intention.get("value", "")

            # 获取观点支持级别
            opinion = self.cognitive_profile.get("opinion", {})
            # 直接从opinions字典中获取viewpoint_N键的值作为支持级别
            viewpoint_1 = self.cognitive_profile['opinion']['viewpoint_1']
            viewpoint_2 = self.cognitive_profile['opinion']['viewpoint_2']
            viewpoint_3 = self.cognitive_profile['opinion']['viewpoint_3']
            viewpoint_4 = self.cognitive_profile['opinion']['viewpoint_4']
            viewpoint_5 = self.cognitive_profile['opinion']['viewpoint_5']
            viewpoint_6 = self.cognitive_profile['opinion']['viewpoint_6']

            # 保存到user_action表
            result = await platform.save_user_action(
                user_id,
                timestep,
                post_id,
                action,
                reason,
                mood_type,
                mood_value,
                emotion_type,
                emotion_value,
                stance_type,
                stance_value,
                thinking_type,
                thinking_value,
                intention_type,
                intention_value,
                "true",
                action_info,
                viewpoint_1,
                viewpoint_2,
                viewpoint_3,
                viewpoint_4,
                viewpoint_5,
                viewpoint_6
            )

            if result.get("success"):
                agent_log.info(f"成功初始化智能体{user_id}的user_action表")
                return True
            else:
                agent_log.warning(f"初始化智能体{user_id}的user_action表失败")
                return False

        except Exception as e:
            agent_log.error(f"初始化智能体{user_id}的user_action表时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            return False

    # 移除think表相关方法

    async def save_initial_state(self):
        """保存智能体的初始状态（轮次为0）

        将智能体的初始认知状态保存为轮次为0的记录

        Returns:
            dict: 包含初始状态的字典
        """
        user_id = self.agent_id

        # 初始状态使用固定值
        action = "initial"
        action_info = json.dumps({}, ensure_ascii=False)
        post_id = 0
        reason = "initial"

        # 提取认知状态中的各项数据
        mood = self.cognitive_profile.get("mood", {})
        mood_type = mood.get("type", "fail")
        mood_value = mood.get("value", "fail")
        # 情绪
        emotion = self.cognitive_profile.get("emotion", {})
        emotion_type = emotion.get("type", "fail")
        emotion_value = emotion.get("value", "fail")
        # 立场
        stance = self.cognitive_profile.get("stance", {})
        stance_type = stance.get("type", "fail")
        stance_value = stance.get("value", "fail")
        # 认知
        thinking = self.cognitive_profile.get("thinking", {})
        thinking_type = thinking.get("type", "fail")
        thinking_value = thinking.get("value", "fail")
        # 意图
        intention = self.cognitive_profile.get("intention", {})
        intention_type = intention.get("type", "fail")
        intention_value = intention.get("value", "fail")
        # 观点
        opinion = self.cognitive_profile.get('opinion', {})
        if isinstance(opinion, list):
            opinion_dict = {}
            for item in opinion:
                for k, v in item.items():
                    if k.startswith("viewpoint_"):
                        opinion_dict[k] = v
            opinion = opinion_dict

        viewpoint_1 = opinion.get("viewpoint_1", "fail")
        viewpoint_2 = opinion.get("viewpoint_2", "fail")
        viewpoint_3 = opinion.get("viewpoint_3", "fail")
        viewpoint_4 = opinion.get("viewpoint_4", "fail")
        viewpoint_5 = opinion.get("viewpoint_5", "fail")
        viewpoint_6 = opinion.get("viewpoint_6", "fail")

        # 创建初始状态字典
        initial_state = {
            "user_id": user_id,
            "user_name": self.user_name,
            "timestep": 0,  # 初始状态的轮次为0
            "is_active": True,  # 初始状态默认为激活
            "post_id": post_id,
            "action": action,
            "mood_type": mood_type,
            "mood_value": mood_value,
            "emotion_type": emotion_type,
            "emotion_value": emotion_value,
            "stance_type": stance_type,
            "stance_value": stance_value,
            "thinking_type": thinking_type,
            "thinking_value": thinking_value,
            "intention_type": intention_type,
            "intention_value": intention_value,
            "viewpoint_1": viewpoint_1,
            "viewpoint_2": viewpoint_2,
            "viewpoint_3": viewpoint_3,
            "viewpoint_4": viewpoint_4,
            "viewpoint_5": viewpoint_5,
            "viewpoint_6": viewpoint_6,
            "action_info": action_info,
            "reason": reason
        }

        # 保存初始状态
        self.initial_state_dict = initial_state
        return initial_state

    async def save_user_action_dict(self, save_mode: str):
        """保存用户行为到数据库或更新user_action_dict

        Args:
            save_mode (str): 保存模式，"db"或"csv"

        Returns:
            bool: 保存是否成功
        """
        try:
            user_id = self.agent_id
            platform = self.env.platform
            timestep = self.step_counter

            # 确保认知状态已初始化
            if not self.cognitive_profile:
                agent_log.warning(f"智能体{user_id}的认知状态未初始化，使用默认值")
                self.cognitive_profile = self.DEFAULT_COGNITIVE_PROFILE.copy()

            # 所有轮次都保存实际用户行为，不再特殊处理第1轮
            if self.is_active:
                # 确保current_action存在且有效
                if not hasattr(self, 'current_action') or not self.current_action:
                    agent_log.warning(f"智能体{user_id}的current_action不存在或为空，使用默认值")
                    self.current_action = {
                        "action_name": "no_action",
                        "arguments": {},
                        "reason": "No action recorded"
                    }

                action = self.current_action.get("action_name", "unknown_action")
                action_info = json.dumps(self.current_action.get("arguments", {}), ensure_ascii=False)
                post_id = getattr(self, 'current_post_id', 0)
                reason = self.current_action.get("reason", "No reason provided")

                # 检查是否是个体模拟模式下的上下文
                if hasattr(self, 'individual_context') and self.individual_context:
                    agent_log.info(f"智能体{user_id}处于个体模拟模式，有上下文数据")
                    # 在个体模拟模式下，确保每个有上下文的用户都生成行为
                    if not action or action == "unknown_action" or action == "no_action":
                        # 如果没有有效的行为，但有上下文，创建一个默认的post行为
                        agent_log.warning(f"智能体{user_id}在个体模拟模式下没有有效行为，创建默认post行为")
                        action = "post"
                        action_info = json.dumps({"content": "Default content for individual simulation"}, ensure_ascii=False)
                        reason = "Default action for individual simulation"
            else:
                # 未激活行为的标准值
                action = "inactive"
                action_info = json.dumps({}, ensure_ascii=False)
                post_id = 0
                reason = "inactive"

            # 提取认知状态中的各项数据
            mood = self.cognitive_profile.get("mood", {})
            mood_type = mood.get("type", "fail")
            mood_value = mood.get("value", "fail")
            # 情绪
            emotion = self.cognitive_profile.get("emotion", {})
            emotion_type = emotion.get("type", "fail")
            emotion_value = emotion.get("value", "fail")
            # 立场
            stance = self.cognitive_profile.get("stance", {})
            stance_type = stance.get("type", "fail")
            stance_value = stance.get("value", "fail")
            # 认知
            thinking = self.cognitive_profile.get("thinking", {})
            thinking_type = thinking.get("type", "fail")
            thinking_value = thinking.get("value", "fail")
            # 意图
            intention = self.cognitive_profile.get("intention", {})
            intention_type = intention.get("type", "fail")
            intention_value = intention.get("value", "fail")
            # 观点
            opinion = self.cognitive_profile.get('opinion', {})
            if isinstance(opinion, list):
                opinion_dict = {}
                for item in opinion:
                    for k, v in item.items():
                        if k.startswith("viewpoint_"):
                            opinion_dict[k] = v
                opinion = opinion_dict

            viewpoint_1 = opinion.get("viewpoint_1", "fail")
            viewpoint_2 = opinion.get("viewpoint_2", "fail")
            viewpoint_3 = opinion.get("viewpoint_3", "fail")
            viewpoint_4 = opinion.get("viewpoint_4", "fail")
            viewpoint_5 = opinion.get("viewpoint_5", "fail")
            viewpoint_6 = opinion.get("viewpoint_6", "fail")

            # 更新user_action_dict
            self.user_action_dict = {
                "user_id": user_id,
                "user_name": self.user_name,
                "timestep": timestep,
                "is_active": self.is_active,
                "post_id": post_id,
                "action": action,
                "mood_type": mood_type,
                "mood_value": mood_value,
                "emotion_type": emotion_type,
                "emotion_value": emotion_value,
                "stance_type": stance_type,
                "stance_value": stance_value,
                "thinking_type": thinking_type,
                "thinking_value": thinking_value,
                "intention_type": intention_type,
                "intention_value": intention_value,
                "viewpoint_1": viewpoint_1,
                "viewpoint_2": viewpoint_2,
                "viewpoint_3": viewpoint_3,
                "viewpoint_4": viewpoint_4,
                "viewpoint_5": viewpoint_5,
                "viewpoint_6": viewpoint_6,
                "action_info": action_info,
                "reason": reason,
            }

            # 记录详细日志
            agent_log.info(f"第{timestep}轮，智能体{user_id} (姓名{self.user_name})的行为: {action}")
            agent_log.debug(f"第{timestep}轮，智能体{user_id}的user_action_dict: {json.dumps(self.user_action_dict, ensure_ascii=False)}")

            if hasattr(self, 'think_content') and self.think_content:
                #agent_log.info(f"第{timestep}轮，智能体{user_id} (姓名{self.user_name})的思考内容：\n{self.think_content}\n")
                # 更新思考内容字典
                self.user_action_dict_think = {
                    "user_id": user_id,
                    "user_name": self.user_name,
                    "timestep": timestep,
                    "is_active": self.is_active,
                    "post_id": post_id,
                    "action": action,
                    "reason": reason,
                    "think": self.think_content
                }

            # 根据保存模式执行不同的操作
            if save_mode == "db":
                # 数据库保存模式：所有数据都保存到数据库
                await platform.save_user_action(
                user_id,
                timestep,
                post_id,
                action,
                reason,
                mood_type,
                mood_value,
                emotion_type,
                emotion_value,
                stance_type,
                stance_value,
                thinking_type,
                thinking_value,
                intention_type,
                intention_value,
                "true" if self.is_active else "false",
                action_info,
                viewpoint_1,
                viewpoint_2,
                viewpoint_3,
                viewpoint_4,
                viewpoint_5,
                viewpoint_6
                )
                agent_log.info(f"【DB保存】已成功保存user_action_dict至数据库 用户ID: {user_id},时间步: {timestep}")
            elif save_mode == "csv":
                # CSV保存模式：只更新user_action_dict，不保存到数据库
                agent_log.info(f"【CSV保存】已成功更新并返回user_action_dict 用户ID: {user_id},时间步: {timestep}")
                return True
            elif save_mode == "both":
                # 混合保存模式：
                # 1. 行为数据（如关注、点赞等）保存到数据库
                # 2. 认知状态只保存在CSV文件中，不保存到数据库的user_action表

                # 创建一个不包含认知状态的行为记录
                # 使用默认值替代认知状态字段
                default_cognitive_value = "not_saved_in_db"

                # 调用平台方法保存行为数据（不包含认知状态）
                await platform.save_user_action(
                user_id,
                timestep,
                post_id,
                action,
                reason,
                default_cognitive_value,  # 不保存认知状态
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                "true" if self.is_active else "false",
                action_info,
                default_cognitive_value,  # 不保存观点
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value,
                default_cognitive_value
                )

                agent_log.info(f"【混合保存】已成功保存行为数据到数据库，认知状态只保存到CSV 用户ID: {user_id},时间步: {timestep}")
                return True
            else:
                agent_log.warning(f"未知的保存模式: {save_mode}")
                return False

            return True

        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 保存user_action_dict出错: {str(e)}")

            import traceback
            agent_log.error(traceback.format_exc())
            return False

    # 移除think表相关方法

    async def initialize_cognitive_profile(self):
        """初始化智能体认知档案"""
        try:
            # 检查用户信息是否完整
            if not hasattr(self, 'user_info') or not hasattr(self.user_info, 'profile') or self.user_info.profile is None:
                agent_log.info(f"Agent {self.agent_id} 信息不完整，创建默认认知档案")
                self.cognitive_profile = self.DEFAULT_COGNITIVE_PROFILE.copy()
                return self.cognitive_profile

            profile = self.user_info.profile

            # 检查是否存在认知档案
            if profile is None or 'other_info' not in profile or profile['other_info'] is None or 'cognitive_profile' not in profile['other_info'] or profile['other_info']['cognitive_profile'] is None:
                agent_log.info(f"Agent {self.agent_id} 数据中未找到有效认知档案，创建默认档案")
                self.cognitive_profile = self.DEFAULT_COGNITIVE_PROFILE.copy()
                return self.cognitive_profile

            # 从用户配置文件中提取认知档案信息
            raw_profile = profile['other_info']['cognitive_profile']
            agent_log.info(f"Agent {self.agent_id} 成功提取原始认知档案")

            # 创建认知档案，确保每个字段都有正确的结构
            self.cognitive_profile = {}

            # 处理基本认知字段
            for field in ['mood', 'emotion', 'stance', 'thinking', 'intention']:
                field_data = raw_profile.get(field, {})
                if not isinstance(field_data, dict):
                    field_data = {}
                self.cognitive_profile[field] = {
                    'type': field_data.get('type', 'fail'),
                    'value': field_data.get('value', 'fail')
                }

            # 添加或更新opinion字段（如果不存在）
            if 'opinion' not in self.cognitive_profile:
                self.cognitive_profile['opinion'] = {}

            # 确保所有观点字段都存在
            for i in range(1, 7):
                viewpoint_key = f'viewpoint_{i}'
                if viewpoint_key not in self.cognitive_profile['opinion']:
                    self.cognitive_profile['opinion'][viewpoint_key] = "fail"

            # 处理观点数据 - 如果原始数据是列表形式
            if isinstance(raw_profile.get('opinion', []), list):
                opinion_list = raw_profile.get('opinion', [])
                # 处理观点列表数据
                for opinion_item in opinion_list:
                    if not isinstance(opinion_item, dict):
                        continue

                    # 提取观点键和支持级别
                    viewpoint_key = None
                    support_level = None

                    for key, value in opinion_item.items():
                        if key.startswith('viewpoint_'):
                            viewpoint_key = key
                            support_level = value  # 在新格式中，viewpoint_x的值直接就是支持级别

                    if viewpoint_key and support_level:
                        normalized_support = self._normalize_support_level(support_level)
                        self.cognitive_profile['opinion'][viewpoint_key] = normalized_support

            # 如果原始数据是字典形式
            elif isinstance(raw_profile.get('opinion', {}), dict):
                opinion_dict = raw_profile.get('opinion', {})
                # 处理观点字典数据
                for key, value in opinion_dict.items():
                    if key.startswith('viewpoint_'):
                        self.cognitive_profile['opinion'][key] = self._normalize_support_level(value)

            agent_log.info(f"Agent {self.agent_id} 认知档案初始化完成")
            return self.cognitive_profile

        except Exception as e:
            agent_log.error(f"Agent {self.agent_id} 初始化认知档案时出错: {str(e)}")
            import traceback
            agent_log.error(traceback.format_exc())
            # 出错时使用默认认知档案
            self.cognitive_profile = self.DEFAULT_COGNITIVE_PROFILE.copy()
            return self.cognitive_profile

    async def save_cognitive_state_to_json(self, output_file_path):
        """
        保存智能体的认知状态、当前行动和提示到JSON文件

        参数:
            output_file_path: 输出文件路径
        """
        # 只有当非inactive状态时才记录
        if self.current_action.get('action_name') == 'inactive':
            return

        # 获取智能体数据
        agent_data = {
            "user_id": self.agent_id,
            "num_steps": self.step_counter,
            "cognitive_state": self.cognitive_profile,
            "current_action": self.current_action,
            "prompt": self.prompt}

        # 加载现有数据或创建新数据结构
        try:
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
        except Exception as e:
            print(f"读取JSON文件时出错: {e}")
            data = []

        # 添加新数据
        data.append(agent_data)

        # 保存回文件
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存JSON文件时出错: {e}")


def save_causal_timing_data(output_dir):
    """
    将全局变量causal_model_timing中的时间数据保存到指定路径的txt文件中
    
    参数:
        output_dir (str): 输出目录路径
    """
    global causal_model_timing
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个因果模型创建单独的txt文件
    for model_name, timing_data in causal_model_timing.items():
        if timing_data:  # 只有当有数据时才创建文件
            output_file = os.path.join(output_dir, f"{model_name}_timing.txt")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    # 写入表头
                    f.write("agent_id\tstep\tduration\ttimestamp\n")
                    
                    # 写入数据
                    for record in timing_data:
                        f.write(f"{record['agent_id']}\t{record['step']}\t{record['duration']:.6f}\t{record['timestamp']}\n")
                
                print(f"因果模型 {model_name} 的时间数据已保存到: {output_file}")
                
            except Exception as e:
                print(f"保存 {model_name} 时间数据时出错: {e}")
        else:
            print(f"因果模型 {model_name} 没有时间数据")
    
    print(f"所有因果模型时间数据保存完成，输出目录: {output_dir}")


def save_causal_numerical_results(output_dir):
    """
    将全局变量causal_model_results中的因果数值结果保存到JSON文件中
    
    参数:
        output_dir (str): 输出目录路径
    """
    global causal_model_results
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个因果模型创建单独的JSON文件
    for model_name, results_data in causal_model_results.items():
        if results_data:  # 只有当有数据时才创建文件
            output_file = os.path.join(output_dir, f"{model_name}_causal_results.json")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, ensure_ascii=False, indent=2)
                
                print(f"因果模型 {model_name} 的数值结果已保存到: {output_file}")
                print(f"  - 包含 {len(results_data)} 次因果分析结果")
                
                # 统计信息
                if results_data:
                    sample_sizes = [r.get('sample_size', 0) for r in results_data]
                    relation_counts = [len(r.get('relations', [])) for r in results_data]
                    print(f"  - 样本量范围: {min(sample_sizes)} - {max(sample_sizes)}")
                    print(f"  - 因果关系数量范围: {min(relation_counts)} - {max(relation_counts)}")
                
            except Exception as e:
                print(f"保存 {model_name} 数值结果时出错: {e}")
        else:
            print(f"因果模型 {model_name} 没有数值结果")
    
    print(f"所有因果模型数值结果保存完成，输出目录: {output_dir}")


def save_all_causal_data(output_dir):
    """
    保存所有因果分析数据（时间数据和数值结果）
    
    参数:
        output_dir (str): 输出目录路径
    """
    print("开始保存因果分析数据...")
    save_causal_timing_data(output_dir)
    save_causal_numerical_results(output_dir)
    print("因果分析数据保存完成！")


