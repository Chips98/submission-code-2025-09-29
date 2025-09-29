"""
思考记录管理模块。
负责记录和管理智能体的思考过程。
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ThinkRecorder:
    """思考记录管理器"""
    
    def __init__(self, agent_id: int, logger: logging.Logger, env: Any):
        """
        初始化思考记录管理器
        
        参数:
            agent_id: 智能体ID
            logger: 日志记录器
            env: 环境对象
        """
        self.agent_id = agent_id
        self.logger = logger
        self.env = env
        self.step_counter = 0
        self.sub_step_counter = 0
        
    async def update_think_record(
        self,
        action_name: str,
        content: str,
        cognitive_state: Optional[Dict] = None,
        reason: Optional[str] = None,
        sub_step: Optional[int] = None,
        post_id: Optional[int] = None
    ) -> bool:
        """
        更新智能体的思考记录
        
        参数:
            action_name: 动作名称
            content: 内容
            cognitive_state: 认知状态
            reason: 动作原因
            sub_step: 子步骤编号
            post_id: 相关帖子ID
            
        返回:
            bool: 更新是否成功
        """
        try:
            # 从环境变量获取当前时间步，如果不存在则使用内部计数器
            try:
                step_number = int(os.environ.get("TIME_STAMP", "0"))
            except (ValueError, TypeError):
                step_number = self.step_counter
            
            # 设置子步骤编号
            if sub_step is None:
                self.sub_step_counter += 1
                sub_step = self.sub_step_counter
            
            # 识别行为类型并标记子步骤值
            if action_name.startswith("timestep_") and action_name.endswith("_end"):
                # 时间步结束标记，使用固定的子步骤值99
                sub_step = 99
                self.sub_step_counter = 0  # 重置子步骤计数器
            
            # 确保action_name不为None且为字符串类型
            if action_name is None:
                action_name = f"timestep_{step_number}_action_{sub_step}"
            elif not isinstance(action_name, str):
                action_name = str(action_name)
                
            # 确保content不为None且为字符串类型
            if content is None:
                content = f"步骤 {step_number} 子步骤 {sub_step} 的内容"
            elif not isinstance(content, str):
                content = str(content)
                
            self.logger.info(
                f"更新思考记录 agent_id={self.agent_id} "
                f"步骤={step_number} "
                f"子步骤={sub_step} "
                f"动作={action_name} "
                f"认知状态={(cognitive_state or {})}"
            )
            
            # 修改参数处理部分
            args = {
                "agent_id": str(self.agent_id),
                "step_number": int(step_number),
                "sub_step_number": int(sub_step),
                "post_id": post_id,
                "action_name": str(action_name),
                "content": str(content),
                "reason": str(reason or ""),
                "cognitive_state": cognitive_state or {}
            }
            
            # 添加环境操作前的完整检查
            if not hasattr(self.env, 'action') or not hasattr(self.env.action, 'update_think'):
                self.logger.error("环境或action未正确初始化")
                return False
            
            # 执行数据库操作
            result = await self.env.action.update_think(**args)
            
            # 添加结果有效性检查
            return bool(result.get("success", False))
                
        except Exception as e:
            self.logger.error(f"更新思考记录出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def increment_step_counter(self):
        """增加步骤计数器"""
        self.step_counter += 1
        
    def reset_sub_step_counter(self):
        """重置子步骤计数器"""
        self.sub_step_counter = 0
        
    def get_step_counter(self) -> int:
        """获取当前步骤计数"""
        return self.step_counter
        
    def get_sub_step_counter(self) -> int:
        """获取当前子步骤计数"""
        return self.sub_step_counter 