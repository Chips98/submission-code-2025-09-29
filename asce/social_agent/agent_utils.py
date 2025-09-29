# -*- coding: utf-8 -*-
"""
社交代理辅助函数和管理器类
包含认知状态管理、模型管理、记忆管理等功能模块
"""
import json
import logging
import os
import re
import asyncio
import aiohttp
import ssl
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage

try:
    from asce.causal.causal_analysis_new import get_causal_relations_for_simulation
except ImportError:
    print("警告: 无法导入因果分析模块")

try:
    from asce.social_agent.prompt.generate_prompt import generate_prompt
except ImportError:
    print("警告: 无法导入prompt生成模块")


def setup_logging(logger_name: str, log_dir: str = None) -> logging.Logger:
    """设置日志记录器，与原版保持一致的格式"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 如果指定了log_dir，在子目录中保存，否则在根log目录保存
        if log_dir:
            log_path = f"./log/{log_dir}"
            os.makedirs(log_path, exist_ok=True)
            file_handler = logging.FileHandler(f"{log_path}/{logger_name}-{now}.log")
        else:
            os.makedirs("./log", exist_ok=True)
            file_handler = logging.FileHandler(f"./log/{logger_name}-{now}.log")
            
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    return logger


def load_cognitive_normalization_map() -> Dict:
    """加载认知规范化映射文件"""
    normalization_map_path = os.path.join(os.path.dirname(__file__), "cognitive_normalization.json")
    try:
        with open(normalization_map_path, 'r', encoding='utf-8') as f:
            normalization_map = json.load(f)
        print(f"成功加载认知规范化映射: {normalization_map_path}")
        return normalization_map
    except Exception as e:
        print(f"加载认知规范化映射失败: {e}，将使用内置默认映射")
        return {}


class CognitiveStateManager:
    """认知状态管理器"""
    
    def __init__(self, normalization_map: Dict = None, validate_state: bool = False, logger: logging.Logger = None):
        """初始化认知状态管理器"""
        self.normalization_map = normalization_map or {}
        self.validate_state = validate_state
        self.cognitive_profile = None
        self.logger = logger or setup_logging("cognitive_manager")
        
        # 默认认知档案
        self.default_profile = {
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

    def extract_cognitive_state(self, content: str) -> Optional[Dict]:
        """从LLM响应中提取认知状态"""
        try:
            # 查找JSON格式的认知状态
            json_pattern = r'\{[^{}]*?"mood"[^{}]*?\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    cognitive_state = json.loads(match)
                    if self._validate_cognitive_structure(cognitive_state):
                        normalized_state = self._normalize_cognitive_state(cognitive_state)
                        return normalized_state
                except json.JSONDecodeError:
                    continue
            
            # 如果没有找到有效的JSON，尝试其他解析方法
            return self._fallback_parse_cognitive_state(content)
            
        except Exception as e:
            self.logger.error(f"提取认知状态失败: {e}")
            return None

    def _validate_cognitive_structure(self, state: Dict) -> bool:
        """验证认知状态结构"""
        required_fields = ['mood', 'emotion', 'stance', 'thinking', 'intention']
        return all(field in state for field in required_fields)

    def _normalize_cognitive_state(self, state: Dict) -> Dict:
        """规范化认知状态"""
        normalized = {}
        
        for field, value in state.items():
            if field in ['mood', 'emotion', 'stance', 'thinking', 'intention']:
                if isinstance(value, dict) and 'type' in value and 'value' in value:
                    normalized[field] = {
                        'type': self._normalize_cognitive_type(field, value['type']),
                        'value': self._normalize_cognitive_value(field, value['type'], value['value'])
                    }
                else:
                    normalized[field] = {'type': 'fail', 'value': 'fail'}
            elif field == 'opinion':
                normalized[field] = self._normalize_opinion(value)
        
        return normalized

    def _normalize_cognitive_type(self, field: str, cog_type: str) -> str:
        """规范化认知类型"""
        if not cog_type or cog_type.strip() == "":
            return "fail"
        
        # 获取该字段的有效类型列表
        field_map = self.normalization_map.get(field, {})
        valid_types = list(field_map.keys())
        
        if not valid_types:
            return cog_type  # 如果没有映射，直接返回原值
        
        # 检查是否已经是有效类型
        if cog_type in valid_types:
            return cog_type
        
        # 尝试模糊匹配
        cog_type_lower = cog_type.lower().strip()
        for valid_type in valid_types:
            if valid_type.lower() in cog_type_lower or cog_type_lower in valid_type.lower():
                return valid_type
        
        return "fail"

    def _normalize_cognitive_value(self, field: str, cog_type: str, cog_value: str) -> str:
        """规范化认知值"""
        if not cog_value or cog_value.strip() == "":
            return "fail"
        
        # 获取该字段和类型的有效值列表
        field_map = self.normalization_map.get(field, {})
        type_map = field_map.get(cog_type, [])
        
        if not type_map:
            return cog_value  # 如果没有映射，直接返回原值
        
        # 检查是否已经是有效值
        if cog_value in type_map:
            return cog_value
        
        # 尝试模糊匹配
        cog_value_lower = cog_value.lower().strip()
        for valid_value in type_map:
            if valid_value.lower() in cog_value_lower or cog_value_lower in valid_value.lower():
                return valid_value
        
        return "fail"

    def _normalize_opinion(self, opinion: Any) -> Dict:
        """规范化观点"""
        if isinstance(opinion, dict):
            normalized_opinion = {}
            for i in range(1, 7):
                key = f'viewpoint_{i}'
                if key in opinion:
                    normalized_opinion[key] = opinion[key]
                else:
                    normalized_opinion[key] = 'fail'
            return normalized_opinion
        else:
            return {f'viewpoint_{i}': 'fail' for i in range(1, 7)}

    def _fallback_parse_cognitive_state(self, content: str) -> Optional[Dict]:
        """后备认知状态解析方法"""
        try:
            # 使用简单的正则表达式尝试提取认知信息
            state = {}
            
            # 提取情绪
            mood_match = re.search(r'情绪[：:]\s*([^，,。.\n]+)', content)
            if mood_match:
                state['mood'] = {'type': 'general', 'value': mood_match.group(1).strip()}
            
            # 提取情感
            emotion_match = re.search(r'情感[：:]\s*([^，,。.\n]+)', content)
            if emotion_match:
                state['emotion'] = {'type': 'general', 'value': emotion_match.group(1).strip()}
            
            # 提取立场
            stance_match = re.search(r'立场[：:]\s*([^，,。.\n]+)', content)
            if stance_match:
                state['stance'] = {'type': 'general', 'value': stance_match.group(1).strip()}
            
            return state if state else None
            
        except Exception as e:
            self.logger.error(f"后备解析失败: {e}")
            return None

    def update_cognitive_profile(self, new_state: Dict):
        """更新认知档案"""
        if not self.cognitive_profile:
            self.cognitive_profile = self.default_profile.copy()
        
        for field, value in new_state.items():
            if field in self.cognitive_profile:
                self.cognitive_profile[field] = value

    def check_cognitive_profile(self) -> bool:
        """检查认知档案是否有效"""
        if not self.cognitive_profile:
            return False
        
        required_fields = ['mood', 'emotion', 'stance', 'thinking', 'intention']
        return all(field in self.cognitive_profile for field in required_fields)


class ModelManager:
    """模型管理器"""
    
    def __init__(
        self,
        model_type: str,
        is_openai_model: bool = False,
        is_deepseek_model: bool = False,
        is_local_model: bool = False,
        deepseek_api_base: str = None,
        local_model_api_base: str = None,
        max_tokens: int = 32000,
        temperature: float = 0.5,
        logger: logging.Logger = None
    ):
        """初始化模型管理器"""
        self.model_type = model_type
        self.is_openai_model = is_openai_model
        self.is_deepseek_model = is_deepseek_model
        self.is_local_model = is_local_model
        self.deepseek_api_base = deepseek_api_base
        self.local_model_api_base = local_model_api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger or setup_logging("model_manager")
        

    async def call_model(self, messages: List, action_handler=None):
        """调用模型API"""
        try:
            return await self._call_local_model(messages)
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            raise

    async def _call_local_model(self, messages: List):
        """调用本地模型API"""
        api_base = self.local_model_api_base or "http://localhost:8889/v1"
        api_base = api_base.rstrip('/')
        
        # 构建OpenAI兼容的消息格式
        openai_messages = []
        for msg in messages:
            if isinstance(msg, str):
                openai_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                openai_messages.append(msg)
            else:
                openai_messages.append({"role": "user", "content": str(msg)})
        
        # 构建请求数据
        data = {
            "model": self.model_type,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # 发送请求
        url = f"{api_base}/chat/completions"
        self.logger.debug(f"发送请求到本地模型API: {url}")
        self.logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)[:500]}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.debug(f"本地模型API响应成功")
                    return result
                else:
                    error_text = await response.text()
                    self.logger.error(f"本地模型API错误响应 {response.status}: {error_text[:500]}")
                    raise Exception(f"本地模型API调用失败: {response.status}")


class MessageProcessor:
    """消息处理器"""
    
    def __init__(self, logger: logging.Logger = None):
        """初始化消息处理器"""
        self.logger = logger or setup_logging("message_processor")

    async def build_user_message(
        self,
        env_prompt: str,
        agent,
        save_mode: str = "db",
        prompt_mode: str = "asce"
    ) -> str:
        """构建用户消息"""
        try:
            # 生成记忆内容
            memory_content = await agent.memory_manager.generate_memory_content(agent.step_counter)
            
            # 构建提示
            try:
                # 准备用户信息
                user_dict = {
                    "agent_id": agent.agent_id,
                    "name": agent.user_info.name if agent.user_info else f"Agent_{agent.agent_id}",
                    "profile": agent.user_info.profile if agent.user_info else {}
                }
                
                user_message = generate_prompt(
                    user=user_dict,
                    env_prompt=env_prompt,
                    cognitive_profile=agent.cognitive_manager.cognitive_profile or {},
                    memory_content=memory_content,
                    action_space_prompt=agent.action_space_prompt,
                    cognition_space_dict=agent.cognition_space_dict,
                    prompt_mode=prompt_mode
                )
            except Exception as e:
                self.logger.warning(f"使用generate_prompt失败: {e}，使用简单提示")
                user_message = f"环境提示: {env_prompt}\n记忆内容: {memory_content}"
            
            return user_message
            
        except Exception as e:
            self.logger.error(f"构建用户消息失败: {e}")
            return env_prompt

    async def validate_messages(self, messages: List[str]) -> List[str]:
        """验证消息格式"""
        valid_messages = []
        
        for message in messages:
            if isinstance(message, str) and len(message.strip()) > 0:
                valid_messages.append(message.strip())
        
        return valid_messages

    def extract_content_from_response(self, response) -> str:
        """从响应中提取内容"""
        try:
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            elif isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    return response['choices'][0]['message']['content']
                elif 'content' in response:
                    return response['content']
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"提取响应内容失败: {e}")
            return ""


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, agent_id: int, csv_path: str, logger: logging.Logger = None):
        """初始化记忆管理器"""
        self.agent_id = agent_id
        self.csv_path = csv_path
        self.logger = logger or setup_logging("memory_manager")
        self.num_historical_memory = 2  # 历史记忆数量
        self.history_memory = "暂无历史记忆。"
        
        # 创建记忆存储目录
        self.memory_dir = os.path.join("memory_data")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_file = os.path.join(self.memory_dir, f"agent_{agent_id}_memory.txt")

    async def generate_memory_content(self, current_timestep: int) -> str:
        """生成记忆内容"""
        try:
            # 如果有CSV路径，从CSV生成记忆
            if self.csv_path and os.path.exists(self.csv_path):
                return await self._generate_memory_from_csv(current_timestep)
            else:
                return await self._generate_memory_from_file(current_timestep)
                
        except Exception as e:
            self.logger.error(f"生成记忆内容失败: {e}")
            return "暂无历史记忆。"

    async def _generate_memory_from_csv(self, current_timestep: int) -> str:
        """从CSV文件生成记忆内容"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # 检查必要的列是否存在
            required_columns = ['user_id', 'timestep']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"CSV文件缺少必要列: {missing_columns}")
                return "暂无历史记忆。"
            
            # 计算历史记忆范围
            start_timestep = max(1, current_timestep - self.num_historical_memory)
            
            # 筛选当前智能体的历史数据 - 使用正确的列名 'user_id'
            agent_data = df[
                (df['user_id'] == self.agent_id) & 
                (df['timestep'] >= start_timestep) & 
                (df['timestep'] < current_timestep)
            ]
            
            if agent_data.empty:
                return "暂无历史记忆。"
            
            # 按时间步排序
            agent_data = agent_data.sort_values(by='timestep', ascending=False)
            
            # 获取最近的记忆条目
            recent_memories = agent_data.head(self.num_historical_memory)
            
            memory_content = "历史记忆:\n"
            for _, row in recent_memories.iterrows():
                timestep = row.get('timestep', '未知')
                action = row.get('action', '未知动作')
                reason = row.get('reason', '未知原因')
                memory_content += f"- 第 {timestep} 步: {action} ({reason})\n"
            
            return memory_content
            
        except Exception as e:
            self.logger.error(f"从CSV生成记忆失败: {e}")
            return "暂无历史记忆。"

    def write_record(self, memory_record):
        """写入记忆记录（兼容CAMEL框架接口）"""
        try:
            # 提取记忆记录中的消息内容
            if hasattr(memory_record, 'message') and hasattr(memory_record.message, 'content'):
                content = memory_record.message.content
            else:
                content = str(memory_record)
            
            # 将记忆写入文件
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Agent {self.agent_id}: {content}\n")
            
            self.logger.debug(f"成功写入记忆记录到 {self.memory_file}")
            
        except Exception as e:
            self.logger.error(f"写入记忆记录失败: {e}")

    async def _generate_memory_from_file(self, current_timestep: int) -> str:
        """从记忆文件生成记忆内容"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content if content.strip() else "暂无历史记忆。"
            else:
                return "暂无历史记忆。"
                
        except Exception as e:
            self.logger.error(f"从文件生成记忆失败: {e}")
            return "暂无历史记忆。"

    async def update_memory(self, timestep: int):
        """更新记忆"""
        try:
            # 这里可以实现记忆更新逻辑
            memory_entry = f"步骤 {timestep}: 完成一轮交互\n"
            
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(memory_entry)
                
        except Exception as e:
            self.logger.error(f"更新记忆失败: {e}")


class DataManager:
    """数据管理器"""
    
    def __init__(self, agent_id: int, data_name: str, logger: logging.Logger = None):
        """初始化数据管理器"""
        self.agent_id = agent_id
        self.data_name = data_name
        self.logger = logger or setup_logging("data_manager")
        
        # 创建数据存储目录
        self.data_dir = os.path.join("data_output")
        os.makedirs(self.data_dir, exist_ok=True)

    def save_user_action_dict(self, action_dict: Dict):
        """保存用户动作字典"""
        try:
            action_file = os.path.join(self.data_dir, f"agent_{self.agent_id}_actions.json")
            
            # 读取现有数据
            if os.path.exists(action_file):
                with open(action_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 添加新数据
            existing_data.append({
                'timestamp': datetime.now().isoformat(),
                'action': action_dict
            })
            
            # 保存数据
            with open(action_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存用户动作字典失败: {e}")

    def save_cognitive_state_to_json(self, cognitive_state: Dict):
        """保存认知状态到JSON"""
        try:
            cognitive_file = os.path.join(self.data_dir, f"agent_{self.agent_id}_cognitive.json")
            
            # 读取现有数据
            if os.path.exists(cognitive_file):
                with open(cognitive_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 添加新数据
            existing_data.append({
                'timestamp': datetime.now().isoformat(),
                'cognitive_state': cognitive_state
            })
            
            # 保存数据
            with open(cognitive_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存认知状态失败: {e}")

    async def save_action_data(self, data: Dict):
        """保存动作数据"""
        try:
            # 保存到通用动作文件
            action_file = os.path.join(self.data_dir, f"agent_{self.agent_id}_all_data.json")
            
            # 读取现有数据
            if os.path.exists(action_file):
                with open(action_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 添加新数据
            existing_data.append(data)
            
            # 保存数据
            with open(action_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存动作数据失败: {e}")

    async def save_timestep_data(self, timestep: int):
        """保存时间步数据"""
        try:
            timestep_file = os.path.join(self.data_dir, f"agent_{self.agent_id}_timesteps.json")
            
            # 读取现有数据
            if os.path.exists(timestep_file):
                with open(timestep_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 添加新数据
            existing_data.append({
                'timestep': timestep,
                'timestamp': datetime.now().isoformat()
            })
            
            # 保存数据
            with open(timestep_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存时间步数据失败: {e}")