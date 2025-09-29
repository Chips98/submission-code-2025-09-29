"""
响应质量统计工具模块

该模块提供了用于打印和记录LLM响应解析质量统计信息的工具函数。
"""

import logging
from asce.social_agent.response_quality import ResponseQualityTracker

# 获取日志记录器
agent_log = logging.getLogger(name="social.agent")

def print_response_quality_stats(save_to_file=True):
    """
    打印LLM响应质量统计信息
    
    该函数获取ResponseQualityTracker单例实例，并调用其print_summary方法
    打印当前的响应质量统计信息。
    
    参数:
        save_to_file: 是否将统计信息保存到文件，默认为True
    """
    try:
        # 获取ResponseQualityTracker单例实例
        quality_tracker = ResponseQualityTracker.get_instance()
        
        # 检查是否启用了统计功能
        if not quality_tracker.enabled:
            agent_log.info("响应质量统计功能已禁用，跳过打印统计信息")
            return
        
        # 打印统计摘要
        quality_tracker.print_summary(save_to_file=save_to_file)
        agent_log.info("已打印响应质量统计信息")
        
    except Exception as e:
        agent_log.error(f"打印响应质量统计信息时出错: {e}")
        import traceback
        agent_log.error(traceback.format_exc())
