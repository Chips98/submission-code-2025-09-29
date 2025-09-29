"""
多API处理器模块

该模块实现了一个处理多个VLLM API端点的类，支持负载均衡和重试机制。
"""

import asyncio
import json
import logging
import random
import aiohttp
from typing import List, Dict, Any, Optional

# 获取日志记录器
social_log = logging.getLogger(name="social")

class MultiApiHandler:
    """
    多API处理器类，用于管理多个VLLM API端点，实现负载均衡和重试机制。

    功能特点:
    1. 支持多个API端点，自动负载均衡
    2. 使用信号量限制每个API的并发请求数
    3. 支持认知状态验证和失败重试
    4. 详细的日志记录
    """

    def __init__(
        self,
        api_urls: List[str],
        max_concurrent_per_api: int = 64,
        max_retries: int = 3,
        validate_cognitive_state: bool = False
    ):
        """
        初始化多API处理器

        Args:
            api_urls: API端点URL列表
            max_concurrent_per_api: 每个API的最大并发请求数
            max_retries: 请求失败时的最大重试次数
            validate_cognitive_state: 是否验证认知状态
        """
        self.api_urls = api_urls
        self.max_concurrent_per_api = max_concurrent_per_api
        self.max_retries = max_retries
        self.validate_cognitive_state = validate_cognitive_state

        # 每个API的调用计数器
        self.api_calls = {url: 0 for url in api_urls}

        # 为每个API创建信号量以控制并发
        self.api_semaphores = {
            url: asyncio.Semaphore(max_concurrent_per_api)
            for url in api_urls
        }

        social_log.info(f"已初始化MultiApiHandler，使用{len(api_urls)}个API端点")
        social_log.info(f"每个API最大并发数: {max_concurrent_per_api}，最大重试次数: {max_retries}")
        social_log.info(f"认知状态验证: {'启用' if validate_cognitive_state else '禁用'}")

    def _select_api(self) -> str:
        """
        选择一个API端点，采用负载最小的策略

        Returns:
            选择的API端点URL
        """
        # 选择调用次数最少的API
        return min(self.api_calls.items(), key=lambda x: x[1])[0]

    def validate_cognitive_state_response(self, response_data: Dict[str, Any]) -> bool:
        """
        验证响应中的认知状态是否符合要求

        Args:
            response_data: API响应数据

        Returns:
            验证结果，True表示通过验证
        """
        if not self.validate_cognitive_state:
            return True

        try:
            # 检查响应中是否包含content字段
            if 'choices' not in response_data or not response_data['choices']:
                social_log.warning("响应中缺少choices字段")
                return False

            content = response_data['choices'][0]['message']['content']

            # 尝试解析JSON内容
            try:
                content_json = json.loads(content)
            except json.JSONDecodeError:
                social_log.warning("响应内容不是有效的JSON格式")
                return False

            # 检查是否包含认知状态字段
            required_dimensions = ['mood', 'emotion', 'thinking', 'stance', 'intention']
            if 'cognitive_profile' not in content_json:
                social_log.warning("响应中缺少cognitive_profile字段")
                return False

            # 检查每个认知维度是否存在
            cognitive_profile = content_json['cognitive_profile']
            for dim in required_dimensions:
                if dim not in cognitive_profile:
                    social_log.warning(f"认知配置中缺少{dim}维度")
                    return False

                # 检查每个维度是否包含type和value字段
                if not isinstance(cognitive_profile[dim], dict):
                    social_log.warning(f"{dim}维度不是字典类型")
                    return False

                if 'type' not in cognitive_profile[dim] or 'value' not in cognitive_profile[dim]:
                    social_log.warning(f"{dim}维度缺少type或value字段")
                    return False

            # 检查是否包含观点字段
            if 'opinion' not in cognitive_profile:
                social_log.warning("认知配置中缺少opinion字段")
                return False

            return True

        except Exception as e:
            social_log.error(f"验证认知状态时出错: {str(e)}")
            return False

    async def call_model_with_retry(self, messages: List[Dict[str, Any]], agent_id: int = None) -> Dict[str, Any]:
        """
        调用模型API，支持重试机制

        Args:
            messages: 要发送给模型的消息列表
            agent_id: 代理ID，用于日志记录

        Returns:
            模型API的响应数据

        Raises:
            Exception: 当所有重试都失败时抛出
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            # 选择一个API端点
            api_url = self._select_api()

            # 增加API调用计数
            self.api_calls[api_url] += 1

            try:
                # 使用信号量控制并发
                async with self.api_semaphores[api_url]:
                    social_log.debug(f"代理{agent_id}: 使用API {api_url}，当前为第{retries}次尝试")

                    # 发送请求到API
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{api_url}/chat/completions",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "llama3.1-8b",  # 使用默认模型名称
                                "messages": messages,
                                "temperature": 0.7,  # 使用默认温度
                                "max_tokens": 4096,  # 使用默认最大token数
                            },
                            timeout=360  # 增加超时时间到120秒
                        ) as response:
                            # 检查响应状态
                            if response.status != 200:
                                error_msg = f"API请求失败，状态码: {response.status}"
                                social_log.warning(f"代理{agent_id}: {error_msg}")
                                last_error = Exception(error_msg)
                                retries += 1
                                continue

                            # 解析响应数据
                            try:
                                response_data = await response.json()

                                # 检查响应是否为空或包含错误
                                if not response_data or (isinstance(response_data, dict) and 'error' in response_data and not response_data.get('choices')):
                                    error_msg = f"API返回空响应或错误: {response_data}"
                                    social_log.warning(f"代理{agent_id}: {error_msg}")
                                    last_error = Exception(error_msg)
                                    retries += 1
                                    # 添加短暂延迟，避免立即重试
                                    await asyncio.sleep(0.5)
                                    continue

                                # 验证认知状态
                                if self.validate_cognitive_state and not self.validate_cognitive_state_response(response_data):
                                    social_log.warning(f"代理{agent_id}: 认知状态验证失败，尝试重试")
                                    retries += 1
                                    # 添加短暂延迟，避免立即重试
                                    await asyncio.sleep(0.5)
                                    continue

                                # 返回成功响应
                                social_log.debug(f"代理{agent_id}: API调用成功")
                                return response_data
                            except Exception as e:
                                error_msg = f"解析API响应时出错: {str(e)}"
                                social_log.warning(f"代理{agent_id}: {error_msg}")
                                last_error = Exception(error_msg)
                                retries += 1
                                continue

            except Exception as e:
                social_log.error(f"代理{agent_id}: API调用出错: {str(e)}")
                last_error = e

            retries += 1

        # 所有重试都失败
        error_msg = f"达到最大重试次数({self.max_retries})，放弃请求"
        social_log.error(f"代理{agent_id}: {error_msg}")
        if last_error:
            raise last_error
        else:
            raise Exception(error_msg)