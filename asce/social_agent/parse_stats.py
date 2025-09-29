"""
解析统计模块

该模块提供了用于收集和报告响应解析统计信息的类。
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

# 获取日志记录器
agent_log = logging.getLogger(name="social.agent")

class ParseStats:
    """
    解析统计类，用于收集和报告响应解析统计信息。
    
    该类跟踪以下统计信息：
    - 总解析尝试次数
    - 成功解析次数
    - 失败解析次数
    - 重试解析次数
    - 每个代理的解析成功率
    - 每种解析策略的使用次数和成功率
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = ParseStats()
        return cls._instance
    
    def __init__(self):
        """初始化统计数据"""
        self.total_attempts = 0
        self.successful_parses = 0
        self.failed_parses = 0
        self.retry_parses = 0
        self.agent_stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0, "retries": 0})
        self.strategy_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
        self.enabled = True
        
    def enable(self):
        """启用统计收集"""
        self.enabled = True
        
    def disable(self):
        """禁用统计收集"""
        self.enabled = False
        
    def reset(self):
        """重置所有统计数据"""
        self.total_attempts = 0
        self.successful_parses = 0
        self.failed_parses = 0
        self.retry_parses = 0
        self.agent_stats.clear()
        self.strategy_stats.clear()
        
    def record_attempt(self, agent_id: int, strategy: str):
        """
        记录解析尝试
        
        Args:
            agent_id: 代理ID
            strategy: 使用的解析策略
        """
        if not self.enabled:
            return
            
        self.total_attempts += 1
        self.agent_stats[agent_id]["attempts"] += 1
        self.strategy_stats[strategy]["attempts"] += 1
        
    def record_success(self, agent_id: int, strategy: str):
        """
        记录解析成功
        
        Args:
            agent_id: 代理ID
            strategy: 使用的解析策略
        """
        if not self.enabled:
            return
            
        self.successful_parses += 1
        self.agent_stats[agent_id]["successes"] += 1
        self.strategy_stats[strategy]["successes"] += 1
        
    def record_failure(self, agent_id: int):
        """
        记录解析失败
        
        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return
            
        self.failed_parses += 1
        self.agent_stats[agent_id]["failures"] += 1
        
    def record_retry(self, agent_id: int):
        """
        记录解析重试
        
        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return
            
        self.retry_parses += 1
        self.agent_stats[agent_id]["retries"] += 1
        
    def get_overall_success_rate(self) -> float:
        """
        获取总体解析成功率
        
        Returns:
            成功率百分比
        """
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_parses / self.total_attempts) * 100
        
    def get_agent_success_rate(self, agent_id: int) -> float:
        """
        获取特定代理的解析成功率
        
        Args:
            agent_id: 代理ID
            
        Returns:
            成功率百分比
        """
        stats = self.agent_stats[agent_id]
        if stats["attempts"] == 0:
            return 0.0
        return (stats["successes"] / stats["attempts"]) * 100
        
    def get_strategy_success_rate(self, strategy: str) -> float:
        """
        获取特定解析策略的成功率
        
        Args:
            strategy: 解析策略
            
        Returns:
            成功率百分比
        """
        stats = self.strategy_stats[strategy]
        if stats["attempts"] == 0:
            return 0.0
        return (stats["successes"] / stats["attempts"]) * 100
        
    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要
        
        Returns:
            包含统计摘要的字典
        """
        return {
            "total_attempts": self.total_attempts,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "retry_parses": self.retry_parses,
            "overall_success_rate": self.get_overall_success_rate(),
            "strategy_stats": {
                strategy: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": self.get_strategy_success_rate(strategy)
                }
                for strategy, stats in self.strategy_stats.items()
            }
        }
        
    def print_summary(self):
        """打印统计摘要"""
        if not self.enabled:
            agent_log.info("解析统计功能已禁用")
            return
            
        summary = self.get_summary()
        
        agent_log.info("=" * 50)
        agent_log.info("响应解析统计摘要")
        agent_log.info("=" * 50)
        agent_log.info(f"总解析尝试次数: {summary['total_attempts']}")
        agent_log.info(f"成功解析次数: {summary['successful_parses']}")
        agent_log.info(f"失败解析次数: {summary['failed_parses']}")
        agent_log.info(f"重试解析次数: {summary['retry_parses']}")
        agent_log.info(f"总体成功率: {summary['overall_success_rate']:.2f}%")
        
        agent_log.info("-" * 50)
        agent_log.info("解析策略统计")
        agent_log.info("-" * 50)
        for strategy, stats in summary["strategy_stats"].items():
            agent_log.info(f"策略 '{strategy}':")
            agent_log.info(f"  尝试次数: {stats['attempts']}")
            agent_log.info(f"  成功次数: {stats['successes']}")
            agent_log.info(f"  成功率: {stats['success_rate']:.2f}%")
            
        agent_log.info("=" * 50)
        
        # 打印到控制台
        print("\n" + "=" * 50)
        print("响应解析统计摘要")
        print("=" * 50)
        print(f"总解析尝试次数: {summary['total_attempts']}")
        print(f"成功解析次数: {summary['successful_parses']}")
        print(f"失败解析次数: {summary['failed_parses']}")
        print(f"重试解析次数: {summary['retry_parses']}")
        print(f"总体成功率: {summary['overall_success_rate']:.2f}%")
        
        print("-" * 50)
        print("解析策略统计")
        print("-" * 50)
        for strategy, stats in summary["strategy_stats"].items():
            print(f"策略 '{strategy}':")
            print(f"  尝试次数: {stats['attempts']}")
            print(f"  成功次数: {stats['successes']}")
            print(f"  成功率: {stats['success_rate']:.2f}%")
            
        print("=" * 50)
