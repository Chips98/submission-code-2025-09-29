"""通用工具函数模块

此模块包含SocialAgent类使用的各种通用工具函数。
"""

import re
import json
import logging
from datetime import datetime
import os

# 配置日志
agent_log = logging.getLogger(name="social.agent")

def extract_reason(content):
    """从内容中提取推理原因"""
    try:
        # 首先检查是否包含```json```格式的JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        
        if json_match:
            # 提取JSON字符串
            json_str = json_match.group(1)
            try:
                # 解析JSON
                parsed_json = json.loads(json_str)
                
                # 检查是否包含reason字段
                if "reason" in parsed_json:
                    reason = parsed_json["reason"]
                    return reason
            except json.JSONDecodeError:
                pass
        
        # 如果没有找到JSON格式的reason，尝试直接解析content为JSON
        try:
            parsed_json = json.loads(content)
            if "reason" in parsed_json:
                reason = parsed_json["reason"]
                return reason
        except json.JSONDecodeError:
            # 尝试使用正则表达式查找reason字段
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
            if reason_match:
                reason = reason_match.group(1)
                return reason
            
            # 尝试提取带引号的reason文本段落
            reason_paragraph_match = re.search(r'"reason"\s*:\s*"(.*?)"[,}]', content, re.DOTALL)
            if reason_paragraph_match:
                reason = reason_paragraph_match.group(1).replace('\n', ' ').strip()
                return reason
        
        # 如果仍然没有找到reason，返回内容的前200个字符作为reason
        if content and isinstance(content, str):
            reason = content[:200].strip()
            return reason
        else:
            return "无推理原因"
    
    except Exception as reason_error:
        import traceback
        return "无推理原因"

def extract_cognitive_state(content, default_state=None):
    """从内容中提取认知状态"""
    # 默认认知状态
    default_cognitive_state = {
        'mood': {'type': 'neutral', 'value': 'neutral'},
        'emotion': {'type': 'neutral', 'value': 'neutral'},
        'stance': {'type': 'neutral', 'value': 'neutral'},
        'thinking': {'type': 'neutral', 'value': 'neutral'},
        'intention': {'type': 'neutral', 'value': 'neutral'}
    }
    
    if default_state is None:
        default_state = default_cognitive_state
    
    # 如果内容为空，返回默认状态
    if not content:
        return default_state
    
    try:
        # 检查是否包含```json```格式的JSON
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        
        if json_match:
            # 提取并解析JSON字符串
            try:
                parsed_json = json.loads(json_match.group(1))
                
                # 如果包含完整的认知状态，直接返回
                if "cognitive_state" in parsed_json and isinstance(parsed_json["cognitive_state"], dict):
                    return parsed_json["cognitive_state"]
                
                # 从各个独立字段中构建认知状态
                cognitive_state = default_state.copy()
                
                for field in ['mood', 'emotion', 'stance', 'thinking', 'intention']:
                    if field in parsed_json:
                        value = parsed_json[field]
                        if isinstance(value, str):
                            cognitive_state[field] = {'type': value.lower(), 'value': value.lower()}
                        elif isinstance(value, dict) and 'type' in value and 'value' in value:
                            cognitive_state[field] = value
                return cognitive_state
            
            except json.JSONDecodeError:
                pass

        # 如果JSON解析失败，尝试使用正则表达式提取
        cognitive_state = default_state.copy()
        
        # 增强字段匹配模式
        field_patterns = {
            'mood': r'(?i)(mood|情感)[：:\s]*([^\s,;]+)',
            'emotion': r'(?i)(emotion|情绪)[：:\s]*([^\s,;]+)',
            'stance': r'(?i)(stance|立场)[：:\s]*([^\s,;]+)',
            'thinking': r'(?i)(thinking|思维)[：:\s]*([^\s,;]+)',
            'intention': r'(?i)(intention|意图)[：:\s]*([^\s,;]+)'
        }
        
        # 添加匹配计数器
        matched_fields = 0
        for field, pattern in field_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(2).lower().strip('.,;!?')
                cognitive_state[field] = {'type': value, 'value': value}
                matched_fields += 1
        
        return cognitive_state
        
    except Exception as e:
        return default_state

def save_to_long_term_memory(memory_file, step_counter, message):
    """将消息保存到长期记忆文件中"""
    try:
        # 确保消息是字符串格式
        if isinstance(message, dict):
            message_str = json.dumps(message, ensure_ascii=False)
        else:
            message_str = str(message)
            
        # 添加时间戳
        timestamped_message = f"[步骤 {step_counter}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{message_str}\n\n"
        
        # 追加写入到文件
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(timestamped_message)
            
        return True
    except Exception as e:
        print(f"保存到长期记忆失败: {str(e)}")
        return False

def get_long_term_memory_summary(memory_file):
    """获取长期记忆的摘要"""
    try:
        if not os.path.exists(memory_file):
            return "没有长期记忆记录。"
            
        # 读取文件内容
        with open(memory_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 如果内容过长，只返回最近的部分
        if len(content) > 2000:
            # 找到最后几个记录的分隔点
            parts = content.split("\n\n")
            recent_parts = parts[-10:]  # 最近的10条记录
            summary = "\n\n".join(recent_parts)
            return f"长期记忆摘要（仅显示最近10条记录）:\n{summary}"
        else:
            return f"长期记忆完整记录:\n{content}"
            
    except Exception as e:
        print(f"获取长期记忆摘要失败: {str(e)}")
        return "无法访问长期记忆。" 