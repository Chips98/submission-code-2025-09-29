# -*- coding: utf-8 -*-
"""
ASCE系统统一日志管理模块
只保留四个核心日志文件，统一存放在时间命名的子目录中
"""
import os
import logging
from datetime import datetime
from typing import Optional


class ASCELoggingSystem:
    """ASCE统一日志系统"""
    
    def __init__(self):
        """初始化日志系统"""
        self.log_dir = None
        self.timestamp = None
        self.loggers = {}
        
    def setup_session_logging(self) -> str:
        """设置会话级别的日志目录"""
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"./log/{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 清理旧的日志处理器
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        self.loggers.clear()
        
        return self.log_dir
    
    def get_agent_logger(self) -> logging.Logger:
        """获取智能体日志器（所有agent共享）"""
        logger_name = "agent_logger"
        if logger_name not in self.loggers:
            logger = logging.getLogger(f"asce.agent.{self.timestamp}")
            logger.setLevel(logging.DEBUG)
            logger.handlers.clear()  # 清除现有处理器
            
            # 创建文件处理器
            handler = logging.FileHandler(
                f"{self.log_dir}/agent_{self.timestamp}.log", 
                encoding="utf-8"
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.propagate = False  # 防止日志传播到根logger
            
            self.loggers[logger_name] = logger
            
        return self.loggers[logger_name]
    
    def get_platform_logger(self) -> logging.Logger:
        """获取平台日志器"""
        logger_name = "platform_logger"
        if logger_name not in self.loggers:
            logger = logging.getLogger(f"asce.platform.{self.timestamp}")
            logger.setLevel(logging.DEBUG)
            logger.handlers.clear()
            
            handler = logging.FileHandler(
                f"{self.log_dir}/platform_{self.timestamp}.log", 
                encoding="utf-8"
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.propagate = False
            
            self.loggers[logger_name] = logger
            
        return self.loggers[logger_name]
    
    def get_simulation_logger(self) -> logging.Logger:
        """获取主模拟流程日志器"""
        logger_name = "simulation_logger"
        if logger_name not in self.loggers:
            logger = logging.getLogger(f"asce.simulation.{self.timestamp}")
            logger.setLevel(logging.DEBUG)
            logger.handlers.clear()
            
            handler = logging.FileHandler(
                f"{self.log_dir}/simulation_{self.timestamp}.log", 
                encoding="utf-8"
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.propagate = False
            
            self.loggers[logger_name] = logger
            
        return self.loggers[logger_name]
    
    def save_config_parameters(self, config_data: dict):
        """保存配置参数到文件"""
        if not self.log_dir:
            self.setup_session_logging()
            
        config_file = f"{self.log_dir}/config_parameters.txt"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("ASCE模拟配置参数\n")
                f.write("=" * 50 + "\n")
                f.write(f"模拟开始时间: {self.timestamp}\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in config_data.items():
                    f.write(f"{key}: {value}\n")
                    
        except Exception as e:
            print(f"保存配置参数失败: {e}")
    
    def cleanup_old_logs(self):
        """清理根目录下的旧日志文件（保持日志目录整洁）"""
        try:
            root_log_dir = "./log"
            if not os.path.exists(root_log_dir):
                return
                
            for filename in os.listdir(root_log_dir):
                file_path = os.path.join(root_log_dir, filename)
                # 删除根目录下的单独日志文件，保留子目录
                if os.path.isfile(file_path) and filename.endswith('.log'):
                    try:
                        os.remove(file_path)
                        print(f"已清理旧日志文件: {filename}")
                    except Exception as e:
                        print(f"清理日志文件失败 {filename}: {e}")
                        
        except Exception as e:
            print(f"清理旧日志文件时出错: {e}")


# 全局单例日志系统
_logging_system = None

def get_logging_system() -> ASCELoggingSystem:
    """获取全局日志系统单例"""
    global _logging_system
    if _logging_system is None:
        _logging_system = ASCELoggingSystem()
    return _logging_system