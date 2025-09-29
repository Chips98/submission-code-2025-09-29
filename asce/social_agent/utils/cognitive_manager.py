"""
认知状态管理模块。
负责初始化、更新和维护智能体的认知状态。
"""

import logging
import json
from typing import Dict, Any, Optional

# 默认认知档案
DEFAULT_COGNITIVE_PROFILE = {
    'mood': {'type': 'neutral', 'value': 'neutral'},
    'emotion': {'type': 'neutral', 'value': 'neutral'},
    'stance': {'type': 'neutral', 'value': 'neutral'},
    'thinking': {'type': 'neutral', 'value': 'neutral'},
    'intention': {'type': 'neutral', 'value': 'neutral'}
}

class CognitiveManager:
    """认知状态管理器"""
    
    def __init__(self, agent_id: int, logger: logging.Logger):
        """
        初始化认知状态管理器
        
        参数:
            agent_id: 智能体ID
            logger: 日志记录器
        """
        self.agent_id = agent_id
        self.logger = logger
        self.cognitive_profile = None
        self.reason = "初始化认知状态"
        
    async def initialize_cognitive_profile(self, user_info: Any) -> Dict:
        """
        初始化智能体的认知档案
        
        参数:
            user_info: 用户信息对象
            
        返回:
            dict: 初始化的认知档案
        """
        try:
            # 检查用户信息是否完整
            if not hasattr(user_info, 'profile'):
                self.logger.debug("用户信息不完整，创建默认认知档案")
                self.cognitive_profile = DEFAULT_COGNITIVE_PROFILE.copy()
                return self.cognitive_profile
            
            profile = user_info.profile
            # 检查是否存在认知档案
            if 'other_info' not in profile or 'cognitive_profile' not in profile['other_info']:
                self.logger.debug("用户数据中未找到认知档案，创建默认档案")
                self.cognitive_profile = DEFAULT_COGNITIVE_PROFILE.copy()
                return self.cognitive_profile
            
            # 从用户配置文件中提取认知档案信息
            raw_profile = profile['other_info']['cognitive_profile']
            
            # 创建认知档案，确保每个字段都有正确的结构
            self.cognitive_profile = {}
            for field in ['mood', 'emotion', 'stance', 'thinking', 'intention']:
                field_data = raw_profile.get(field, {})
                if not isinstance(field_data, dict):
                    field_data = {}
                
                self.cognitive_profile[field] = {
                    'type': field_data.get('type', 'neutral'),
                    'value': field_data.get('value', 'neutral')
                }
            
            self.logger.debug(f"初始化后的认知档案: {self.cognitive_profile}")
            return self.cognitive_profile
            
        except Exception as e:
            self.logger.error(f"初始化认知档案时出错: {str(e)}")
            # 出错时使用默认档案
            self.cognitive_profile = DEFAULT_COGNITIVE_PROFILE.copy()
            return self.cognitive_profile
            
    def update_cognitive_profile(self, new_state: Optional[Dict] = None) -> Dict:
        """
        更新认知档案
        
        参数:
            new_state: 新的认知状态，如果为None则使用默认状态
            
        返回:
            dict: 更新后的认知档案
        """
        if new_state is None:
            new_state = DEFAULT_COGNITIVE_PROFILE.copy()
            
        # 验证新状态的完整性
        required_fields = ['mood', 'emotion', 'stance', 'thinking', 'intention']
        for field in required_fields:
            if field not in new_state:
                new_state[field] = DEFAULT_COGNITIVE_PROFILE[field].copy()
            if not isinstance(new_state[field], dict):
                new_state[field] = DEFAULT_COGNITIVE_PROFILE[field].copy()
            if 'type' not in new_state[field] or 'value' not in new_state[field]:
                new_state[field] = DEFAULT_COGNITIVE_PROFILE[field].copy()
                
        self.cognitive_profile = new_state
        return self.cognitive_profile
        
    def get_cognitive_profile(self) -> Dict:
        """
        获取当前认知档案
        
        返回:
            dict: 当前认知档案
        """
        if self.cognitive_profile is None:
            self.cognitive_profile = DEFAULT_COGNITIVE_PROFILE.copy()
        return self.cognitive_profile
        
    def set_reason(self, reason: str):
        """
        设置认知状态变化的原因
        
        参数:
            reason: 变化原因
        """
        self.reason = reason
        
    def get_reason(self) -> str:
        """
        获取认知状态变化的原因
        
        返回:
            str: 变化原因
        """
        return self.reason 