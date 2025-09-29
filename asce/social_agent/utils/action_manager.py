"""
行为状态管理模块。
负责初始化、更新和维护智能体的行为状态。
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# 默认行为信息
DEFAULT_ACTION_INFO = {
    'action': 'no_action',
    'reason': '初始化状态',
    'post_id': None,
    'is_active': 'no_active'
}

class ActionManager:
    """行为状态管理器"""
    
    def __init__(self, agent_id: int, logger: logging.Logger, env: Any = None):
        """
        初始化行为状态管理器
        
        参数:
            agent_id: 智能体ID
            logger: 日志记录器
            env: 环境对象
        """
        self.agent_id = agent_id
        self.logger = logger
        self.env = env
        self.current_action = None
        self.action_history = []
        self.last_active_status = "no_active"
        
    async def initialize_action_information(self, user_info: Any) -> Dict:
        """
        初始化智能体的行为信息
        
        参数:
            user_info: 用户信息对象
            
        返回:
            dict: 初始化的行为信息
        """
        try:
            # 检查用户信息是否完整
            if not hasattr(user_info, 'profile'):
                self.logger.debug("用户信息不完整，创建默认行为信息")
                self.current_action = DEFAULT_ACTION_INFO.copy()
                return self.current_action
            
            # 从用户信息中提取行为倾向或使用默认值
            profile = user_info.profile
            
            # 如果用户配置文件中有行为倾向信息，则使用它们
            action_info = DEFAULT_ACTION_INFO.copy()
            if 'other_info' in profile and 'action_tendency' in profile['other_info']:
                action_tendency = profile['other_info']['action_tendency']
                if isinstance(action_tendency, dict) and 'preferred_action' in action_tendency:
                    action_info['action'] = action_tendency['preferred_action']
                    action_info['reason'] = action_tendency.get('reason', '基于用户配置的行为偏好')
            
            self.current_action = action_info
            self.logger.debug(f"初始化后的行为信息: {self.current_action}")
            
            # 添加到历史记录
            self.action_history.append({
                **self.current_action,
                "timestamp": datetime.now().isoformat()
            })
            
            return self.current_action
            
        except Exception as e:
            self.logger.error(f"初始化行为信息时出错: {str(e)}")
            # 出错时使用默认行为信息
            self.current_action = DEFAULT_ACTION_INFO.copy()
            return self.current_action
            
    def update_action_info(self, action: str = None, reason: str = None, post_id: int = None, is_active: str = None) -> Dict:
        """
        更新行为信息
        
        参数:
            action: 行为类型
            reason: 行为原因
            post_id: 相关帖子ID
            is_active: 是否激活
            
        返回:
            dict: 更新后的行为信息
        """
        try:
            # 如果当前没有行为信息，则使用默认值
            if self.current_action is None:
                self.current_action = DEFAULT_ACTION_INFO.copy()
            
            # 更新行为信息（只更新非None的字段）
            if action is not None:
                self.current_action['action'] = action
            if reason is not None:
                self.current_action['reason'] = reason
            if post_id is not None:
                self.current_action['post_id'] = post_id
            if is_active is not None:
                self.current_action['is_active'] = is_active
                self.last_active_status = is_active
                
            # 添加到历史记录
            self.action_history.append({
                **self.current_action,
                "timestamp": datetime.now().isoformat()
            })
            
            return self.current_action
        except Exception as e:
            self.logger.error(f"更新行为信息时出错: {str(e)}")
            return self.current_action or DEFAULT_ACTION_INFO.copy()
    
    def record_user_action(self, action: str, reason: str, post_id: int = None) -> Dict:
        """
        记录用户执行的行为
        
        参数:
            action: 行为类型
            reason: 行为原因
            post_id: 相关帖子ID
            
        返回:
            dict: 记录的行为信息
        """
        try:
            # 更新当前行为信息
            self.current_action = {
                'action': action,
                'reason': reason,
                'post_id': post_id,
                'is_active': 'active'  # 如果用户执行了行为，那么肯定是被激活的
            }
            
            # 添加到历史记录
            self.action_history.append({
                **self.current_action,
                "timestamp": datetime.now().isoformat()
            })
            
            return self.current_action
        except Exception as e:
            self.logger.error(f"记录用户行为时出错: {str(e)}")
            return DEFAULT_ACTION_INFO.copy()
    
    def get_current_action(self) -> Dict:
        """
        获取当前行为信息
        
        返回:
            dict: 当前行为信息
        """
        if self.current_action is None:
            self.current_action = DEFAULT_ACTION_INFO.copy()
        return self.current_action
    
    def get_action_history(self) -> List[Dict]:
        """
        获取行为历史记录
        
        返回:
            list: 行为历史记录列表
        """
        return self.action_history
    
    def set_active_status(self, is_active: Any) -> None:
        """
        设置激活状态
        
        参数:
            is_active: 是否激活，可以是布尔值或其他类型
        """
        # 确保参数为布尔值
        if isinstance(is_active, bool):
            status = "active" if is_active else "no_active"
        elif isinstance(is_active, (int, float)):
            # 处理数值型参数
            status = "active" if bool(is_active) else "no_active"
        elif isinstance(is_active, str):
            # 处理字符串参数
            status = "active" if is_active.lower() in ["true", "active", "yes", "1"] else "no_active"
        else:
            # 默认情况
            self.logger.warning(f"未识别的激活状态类型: {type(is_active)}, 值: {is_active}, 默认为'no_active'")
            status = "no_active"
            
        self.last_active_status = status
        
        # 如果当前没有行为信息，则创建一个
        if self.current_action is None:
            self.current_action = DEFAULT_ACTION_INFO.copy()
        
        # 更新激活状态
        self.current_action['is_active'] = status
        
        # 如果未激活，则将行为设置为no_action
        if status != "active":
            self.current_action['action'] = 'no_action'
            self.current_action['reason'] = '用户在此轮未被激活'
    
    def get_active_status(self) -> str:
        """
        获取激活状态
        
        返回:
            str: 激活状态
        """
        return self.last_active_status
