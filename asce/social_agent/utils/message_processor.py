"""消息处理模块

此模块包含处理和验证消息的相关函数。
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

# 配置日志
agent_log = logging.getLogger(name="social.agent")

def validate_messages(openai_messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """验证消息格式，尽量修正消息结构，而不是跳过"""
    valid_messages = []
    
    for idx, msg in enumerate(openai_messages):
        # 如果不是字典，尝试将其转为字典
        if not isinstance(msg, dict):
            print(f"警告: 消息[{idx}]不是字典类型，尝试跳过: {msg}")
            continue  # 非字典实在无法修正
        
        # 如果缺少 role 字段，尝试补充为 'user'
        if "role" not in msg:
            print(f"警告: 消息[{idx}]没有role字段，默认补充为'user': {msg}")
            msg["role"] = "user"  # 默认设为 user
        
        # 如果缺少 content 字段，补空字符串
        if "content" not in msg:
            print(f"警告: 消息[{idx}]没有content字段，已补充空字符串")
            msg["content"] = ""
        
        # content 为 None，替换为空字符串
        if msg["content"] is None:
            print(f"警告: 消息[{idx}]的content字段为None，已替换为空字符串")
            msg["content"] = ""
        
        # content 不是字符串，强制转为字符串
        if not isinstance(msg["content"], str):
            original_type = type(msg["content"]).__name__
            msg["content"] = str(msg["content"])
            print(f"警告: 消息[{idx}]的content字段不是字符串类型(是{original_type})，已转换为字符串")
        
        # role 字段不是标准值，提醒
        if msg["role"] not in ["system", "user", "assistant", "tool"]:
            print(f"警告: 消息[{idx}]的role字段值 '{msg['role']}' 可能不符合OpenAI规范")
        
        valid_messages.append(msg)
    
    # 如果全部跳过，仍然提醒
    if not valid_messages:
        print("错误: 没有有效的消息可以发送给模型")
        return None
    
    return valid_messages


def prepare_deepseek_messages(valid_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为DeepSeek模型准备消息"""
    for i, msg in enumerate(valid_messages):
        # 确保role字段是DeepSeek支持的值
        if msg["role"] not in ["user", "assistant", "system"]:
            original_role = msg["role"]
            if msg["role"].lower() == "user":
                valid_messages[i]["role"] = "user"
            elif msg["role"].lower() == "assistant":
                valid_messages[i]["role"] = "assistant"
            else:
                valid_messages[i]["role"] = "user"  # 默认设为user
            print(f"警告: 消息的role字段'{original_role}'不被DeepSeek支持，已转换为'{valid_messages[i]['role']}'")
    
    return valid_messages

def extract_functions(content: str) -> Optional[Dict[str, Any]]:
    """从内容中提取函数调用信息"""
    try:
        # 检查是否包含```json```格式的JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        
        if json_match:
            # 提取并解析JSON字符串
            try:
                parsed_json = json.loads(json_match.group(1))
                if "functions" in parsed_json and parsed_json["functions"]:
                    return parsed_json["functions"][0]
            except json.JSONDecodeError:
                agent_log.warning("无法解析JSON代码块")
        
        # 尝试直接解析content为JSON
        try:
            parsed_json = json.loads(content)
            if "functions" in parsed_json and parsed_json["functions"]:
                return parsed_json["functions"][0]
        except json.JSONDecodeError:
            # 使用正则表达式查找functions字段
            functions_match = re.search(r'"functions"\s*:\s*\[\s*{\s*"name"\s*:\s*"([^"]+)"', content)
            if functions_match:
                return {"name": functions_match.group(1), "arguments": {}}
    
    except Exception as e:
        agent_log.error(f"提取函数调用信息错误: {e}")
    
    return None 