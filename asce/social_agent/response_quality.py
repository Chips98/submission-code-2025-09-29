"""
响应质量记录模块

该模块提供了用于记录LLM响应解析质量的类。
"""

import logging
from typing import Dict, Any, Optional
from collections import defaultdict

# 获取日志记录器
agent_log = logging.getLogger(name="social.agent")

class ResponseQualityTracker:
    """
    响应质量跟踪器，用于记录LLM响应解析质量。

    该类跟踪以下统计信息：
    - 总响应次数
    - 首次成功解析次数
    - 重试后成功解析次数
    - 解析失败次数
    - 每个代理的解析成功率
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = ResponseQualityTracker()
        return cls._instance

    def __init__(self):
        """初始化统计数据"""
        self.total_responses = 0
        self.first_attempt_success = 0
        self.retry_success = 0
        self.parse_failure = 0
        self.agent_stats = defaultdict(lambda: {"total": 0, "first_success": 0, "retry_success": 0, "failure": 0})
        self.enabled = True

    def enable(self):
        """启用统计收集"""
        self.enabled = True

    def disable(self):
        """禁用统计收集"""
        self.enabled = False

    def reset(self):
        """重置所有统计数据"""
        self.total_responses = 0
        self.first_attempt_success = 0
        self.retry_success = 0
        self.parse_failure = 0
        self.agent_stats.clear()

    def record_response(self, agent_id: int):
        """
        记录响应尝试

        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return

        self.total_responses += 1
        self.agent_stats[agent_id]["total"] += 1

    def record_first_success(self, agent_id: int):
        """
        记录首次尝试成功

        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return

        self.first_attempt_success += 1
        self.agent_stats[agent_id]["first_success"] += 1

    def record_retry_success(self, agent_id: int):
        """
        记录重试成功

        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return

        self.retry_success += 1
        self.agent_stats[agent_id]["retry_success"] += 1

    def record_failure(self, agent_id: int):
        """
        记录解析失败

        Args:
            agent_id: 代理ID
        """
        if not self.enabled:
            return

        self.parse_failure += 1
        self.agent_stats[agent_id]["failure"] += 1

    def get_overall_success_rate(self) -> float:
        """
        获取总体解析成功率

        Returns:
            成功率百分比
        """
        if self.total_responses == 0:
            return 0.0
        return ((self.first_attempt_success + self.retry_success) / self.total_responses) * 100

    def get_first_attempt_success_rate(self) -> float:
        """
        获取首次尝试成功率

        Returns:
            成功率百分比
        """
        if self.total_responses == 0:
            return 0.0
        return (self.first_attempt_success / self.total_responses) * 100

    def get_retry_success_rate(self) -> float:
        """
        获取重试成功率

        Returns:
            成功率百分比
        """
        if self.total_responses == 0:
            return 0.0
        return (self.retry_success / self.total_responses) * 100

    def get_failure_rate(self) -> float:
        """
        获取失败率

        Returns:
            失败率百分比
        """
        if self.total_responses == 0:
            return 0.0
        return (self.parse_failure / self.total_responses) * 100

    def get_agent_success_rate(self, agent_id: int) -> Dict[str, float]:
        """
        获取特定代理的解析成功率

        Args:
            agent_id: 代理ID

        Returns:
            包含各种成功率的字典
        """
        stats = self.agent_stats[agent_id]
        total = stats["total"]

        if total == 0:
            return {
                "overall": 0.0,
                "first_attempt": 0.0,
                "retry": 0.0,
                "failure": 0.0
            }

        return {
            "overall": ((stats["first_success"] + stats["retry_success"]) / total) * 100,
            "first_attempt": (stats["first_success"] / total) * 100,
            "retry": (stats["retry_success"] / total) * 100,
            "failure": (stats["failure"] / total) * 100
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要

        Returns:
            包含统计摘要的字典
        """
        return {
            "total_responses": self.total_responses,
            "first_attempt_success": self.first_attempt_success,
            "retry_success": self.retry_success,
            "parse_failure": self.parse_failure,
            "overall_success_rate": self.get_overall_success_rate(),
            "first_attempt_success_rate": self.get_first_attempt_success_rate(),
            "retry_success_rate": self.get_retry_success_rate(),
            "failure_rate": self.get_failure_rate()
        }

    def print_summary(self, save_to_file=True):
        """打印统计摘要

        参数:
            save_to_file: 是否保存到文件
        """
        if not self.enabled:
            agent_log.info("响应质量统计功能已禁用")
            return

        summary = self.get_summary()

        # 构建摘要文本
        summary_text = []
        summary_text.append("=" * 50)
        summary_text.append("LLM响应质量统计摘要")
        summary_text.append("=" * 50)
        summary_text.append(f"总响应次数: {summary['total_responses']}")
        summary_text.append(f"首次成功解析次数: {summary['first_attempt_success']}")
        summary_text.append(f"重试后成功解析次数: {summary['retry_success']}")
        summary_text.append(f"解析失败次数: {summary['parse_failure']}")
        summary_text.append(f"总体成功率: {summary['overall_success_rate']:.2f}%")
        summary_text.append(f"首次尝试成功率: {summary['first_attempt_success_rate']:.2f}%")
        summary_text.append(f"重试成功率: {summary['retry_success_rate']:.2f}%")
        summary_text.append(f"失败率: {summary['failure_rate']:.2f}%")

        summary_text.append("-" * 50)
        summary_text.append("代理解析统计")
        summary_text.append("-" * 50)
        for agent_id, stats in self.agent_stats.items():
            rates = self.get_agent_success_rate(agent_id)
            summary_text.append(f"代理 {agent_id}:")
            summary_text.append(f"  总响应次数: {stats['total']}")
            summary_text.append(f"  首次成功解析次数: {stats['first_success']}")
            summary_text.append(f"  重试后成功解析次数: {stats['retry_success']}")
            summary_text.append(f"  解析失败次数: {stats['failure']}")
            summary_text.append(f"  总体成功率: {rates['overall']:.2f}%")
            summary_text.append(f"  首次尝试成功率: {rates['first_attempt']:.2f}%")
            summary_text.append(f"  重试成功率: {rates['retry']:.2f}%")
            summary_text.append(f"  失败率: {rates['failure']:.2f}%")

        summary_text.append("=" * 50)

        # 将摘要文本转换为字符串
        summary_str = "\n".join(summary_text)

        # 记录到日志
        for line in summary_text:
            agent_log.info(line)

        # 打印到控制台
        #print("\n" + summary_str)

        # 保存到文件
        if save_to_file:
            self._save_summary_to_file(summary_str)

    def _save_summary_to_file(self, summary_text):
        """将摘要保存到文件

        参数:
            summary_text: 摘要文本
        """
        try:
            import os
            import datetime

            # 创建logs目录（如果不存在）
            logs_dir = "response_quality_logs"
            os.makedirs(logs_dir, exist_ok=True)

            # 获取当前时间
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # 构建文件名
            filename = os.path.join(logs_dir, f"response_quality_{timestamp}.txt")

            # 写入文件
            with open(filename, "w", encoding="utf-8") as f:
                f.write(summary_text)

            agent_log.info(f"响应质量统计摘要已保存到文件: {filename}")

        except Exception as e:
            agent_log.error(f"保存响应质量统计摘要到文件时出错: {e}")
            import traceback
            agent_log.error(traceback.format_exc())
