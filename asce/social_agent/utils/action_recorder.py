"""
行为记录管理模块。
负责记录和管理智能体的行为记录。
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3
import time
import json

class ActionRecorder:
    """行为记录管理器"""
    
    def __init__(self, agent_id: int, logger: logging.Logger, env: Any):
        """
        初始化行为记录管理器
        
        参数:
            agent_id: 智能体ID
            logger: 日志记录器
            env: 环境对象
        """
        self.agent_id = agent_id
        self.logger = logger
        self.env = env
        self.step_counter = 0
        
    async def update_action_record(
        self,
        action: str,
        reason: str,
        post_id: Optional[int] = None,
        cognitive_state: Optional[Dict] = None,
        is_active: str = "no_active",
        num_steps: Optional[int] = None
    ) -> bool:
        """
        更新智能体的行为记录
        
        参数:
            action: 行为类型
            reason: 行为原因
            post_id: 相关帖子ID
            cognitive_state: 认知状态
            is_active: 激活状态
            num_steps: 时间步编号
            
        返回:
            bool: 更新是否成功
        """
        try:
            # 从环境变量获取当前时间步，如果不存在则使用内部计数器或指定的num_steps
            if num_steps is not None:
                step_number = num_steps
            else:
                try:
                    step_number = int(os.environ.get("TIME_STAMP", "0"))
                except (ValueError, TypeError):
                    step_number = self.step_counter
            
            # 确保action不为None且为字符串类型
            if action is None:
                action = "no_action"
            elif not isinstance(action, str):
                action = str(action)
                
            # 确保reason不为None且为字符串类型
            if reason is None:
                reason = "未知原因"
            elif not isinstance(reason, str):
                reason = str(reason)
                
            self.logger.info(
                f"更新行为记录 agent_id={self.agent_id} "
                f"步骤={step_number} "
                f"行为={action} "
                f"原因={reason} "
                f"激活状态={is_active}"
            )
            
            # 准备数据参数
            args = {
                "agent_id": self.agent_id,
                "num_steps": step_number,
                "post_id": post_id,
                "action": action,
                "reason": reason,
                "cognitive_state": cognitive_state or {},
                "is_active": is_active
            }
            
            # 直接使用_direct_db_update方法，不再尝试调用可能不存在的外部方法
            self.logger.debug("直接调用_direct_db_update方法更新数据库")
            return await self._direct_db_update(**args)
                
        except Exception as e:
            self.logger.error(f"更新行为记录出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    async def _direct_db_update(self, **kwargs):
        """
        直接更新数据库中的行为表。
        
        参数:
            **kwargs: 包含更新所需的所有参数。必须包含以下键：
                - agent_id: 代理ID
                - timestep: 时间步
                - action_info: 动作信息
                - cognitive_info: 认知状态信息
                
        返回:
            bool: 操作是否成功
        """
        start_time = time.time()
        self.logger.info(f"开始直接更新数据库，agent_id={kwargs.get('agent_id')}, timestep={kwargs.get('timestep')}")
        self.logger.debug(f"更新参数: {kwargs}")
        
        # 检查必要参数
        required_params = ['agent_id', 'timestep', 'action_info', 'cognitive_info']
        for param in required_params:
            if param not in kwargs:
                self.logger.error(f"缺少必要参数: {param}")
                return False
        
        try:
            # 检查是否有pl_utils属性
            db_cursor = None
            db_connection = None
            
            if hasattr(self, 'pl_utils') and self.pl_utils is not None:
                try:
                    db_cursor = self.pl_utils.db_cursor
                    db_connection = self.pl_utils.db
                    self.logger.debug(f"使用直接设置的pl_utils属性获取数据库连接: cursor={db_cursor}, connection={db_connection}")
                except Exception as e:
                    self.logger.warning(f"从pl_utils获取数据库连接失败: {str(e)}")
                    
            # 如果没有pl_utils或获取失败，尝试从环境中查找
            if db_cursor is None or db_connection is None:
                self.logger.debug("尝试从环境中查找数据库连接")
                
                # 尝试不同路径查找数据库连接
                possible_paths = [
                    ('self.env', lambda: self.env),
                    ('self.env.platform', lambda: self.env.platform),
                    ('self.env.platform.pl_utils', lambda: self.env.platform.pl_utils),
                    ('self.env.action.channel', lambda: self.env.action.channel),
                    ('self.env.action.channel.platform', lambda: self.env.action.channel.platform),
                    ('self.env.action.channel.platform.pl_utils', lambda: self.env.action.channel.platform.pl_utils),
                    ('self.env.action.channel.pl_utils', lambda: self.env.action.channel.pl_utils)
                ]
                
                for path_name, path_func in possible_paths:
                    try:
                        obj = path_func()
                        if hasattr(obj, 'pl_utils'):
                            pl_utils = obj.pl_utils
                            self.logger.debug(f"在{path_name}.pl_utils找到数据库工具")
                            db_cursor = pl_utils.db_cursor
                            db_connection = pl_utils.db
                            if db_cursor and db_connection:
                                self.logger.info(f"成功从{path_name}.pl_utils获取数据库连接")
                                # 缓存找到的pl_utils以便下次使用
                                self.pl_utils = pl_utils
                                break
                        elif hasattr(obj, 'db_cursor') and hasattr(obj, 'db'):
                            self.logger.debug(f"在{path_name}找到数据库连接")
                            db_cursor = obj.db_cursor
                            db_connection = obj.db
                            if db_cursor and db_connection:
                                self.logger.info(f"成功从{path_name}获取数据库连接")
                                break
                    except Exception as e:
                        self.logger.debug(f"尝试路径{path_name}失败: {str(e)}")
            
            # 如果仍然无法获取数据库连接，记录错误并返回
            if db_cursor is None or db_connection is None:
                self.logger.error("无法获取数据库连接，更新失败")
                return False
            
            # 记录当前时间
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 查询用户ID
            agent_id = kwargs['agent_id']
            try:
                db_cursor.execute("SELECT id FROM user WHERE agent_id = ?", (agent_id,))
                user_row = db_cursor.fetchone()
                if user_row:
                    user_id = user_row[0]
                    self.logger.debug(f"找到用户ID: {user_id}，对应代理ID: {agent_id}")
                else:
                    self.logger.warning(f"未找到代理ID {agent_id} 对应的用户ID，将使用代理ID作为用户ID")
                    user_id = agent_id
            except Exception as e:
                self.logger.warning(f"查询用户ID时出错: {str(e)}，将使用代理ID作为用户ID")
                user_id = agent_id
            
            # 准备要插入或更新的数据
            timestep = kwargs['timestep']
            action_info = json.dumps(kwargs['action_info']) if isinstance(kwargs['action_info'], dict) else kwargs['action_info']
            cognitive_info = json.dumps(kwargs['cognitive_info']) if isinstance(kwargs['cognitive_info'], dict) else kwargs['cognitive_info']
            
            # 检查记录是否已存在
            try:
                db_cursor.execute(
                    "SELECT id FROM user_action WHERE user_id = ? AND timestep = ?",
                    (user_id, timestep)
                )
                existing_record = db_cursor.fetchone()
                
                if existing_record:
                    # 更新现有记录
                    record_id = existing_record[0]
                    self.logger.debug(f"找到现有记录ID: {record_id}，将进行更新")
                    
                    update_query = """
                    UPDATE user_action 
                    SET action_info = ?, cognitive_info = ?, updated_at = ? 
                    WHERE id = ?
                    """
                    
                    db_cursor.execute(update_query, (action_info, cognitive_info, current_time, record_id))
                    self.logger.info(f"已更新用户行为记录: id={record_id}, user_id={user_id}, timestep={timestep}")
                else:
                    # 插入新记录
                    self.logger.debug(f"未找到现有记录，将创建新记录: user_id={user_id}, timestep={timestep}")
                    
                    insert_query = """
                    INSERT INTO user_action (user_id, timestep, action_info, cognitive_info, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
                    
                    db_cursor.execute(insert_query, (user_id, timestep, action_info, cognitive_info, current_time, current_time))
                    record_id = db_cursor.lastrowid
                    self.logger.info(f"已创建新的用户行为记录: id={record_id}, user_id={user_id}, timestep={timestep}")
                
                # 提交事务
                db_connection.commit()
                
                end_time = time.time()
                self.logger.debug(f"数据库更新完成，耗时: {end_time - start_time:.3f}秒")
                return True
                
            except sqlite3.Error as e:
                self.logger.error(f"数据库操作失败: {str(e)}")
                if db_connection:
                    db_connection.rollback()
                return False
                
        except Exception as e:
            self.logger.error(f"更新数据库时发生异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def increment_step_counter(self):
        """增加步骤计数器"""
        self.step_counter += 1
        
    def get_step_counter(self) -> int:
        """获取当前步骤计数"""
        return self.step_counter 