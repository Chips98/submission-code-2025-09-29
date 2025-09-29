"""模型处理模块

此模块包含处理不同模型调用和响应的相关函数。
"""

import json
import logging
import aiohttp
import re
from typing import Dict, Any, Optional, Union

# 配置日志
agent_log = logging.getLogger(name="social.agent")

async def call_local_model_api(messages: list, api_base: str = "http://localhost:8889/v1") -> Dict[str, Any]:
    """调用本地模型API"""
    try:
        # 确保API基础URL正确
        api_base = api_base.rstrip('/')
        
        # 构建请求URL
        url = f"{api_base}/chat/completions"
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 构建请求数据
        data = {
            "model": "llama3.1-8b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        # 发送请求
        agent_log.debug(f"发送请求到本地模型API: {url}")
        agent_log.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        
        # 使用aiohttp发送异步请求
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()
                agent_log.debug(f"本地模型API响应: {json.dumps(response_json, ensure_ascii=False)}")
                return response_json
                
    except Exception as e:
        agent_log.error(f"调用本地模型API失败: {str(e)}")
        import traceback
        agent_log.error(traceback.format_exc())
        return {"error": str(e)}

async def process_openai_response(message: Any) -> str:
    """处理OpenAI模型的响应"""
    try:
        # 获取消息内容
        if hasattr(message, 'content'):
            content = message.content
        else:
            content = str(message)
            
        agent_log.debug(f"OpenAI响应内容: {content[:100]}..." if len(content) > 100 else f"OpenAI响应内容: {content}")
        
        # 如果包含tool_calls，优先处理
        if hasattr(message, 'tool_calls') and message.tool_calls:
            agent_log.debug(f"发现tool_calls: {len(message.tool_calls)}")
            
            # 尝试从内容中提取reason
            if content and isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                    if "reason" in parsed_content:
                        reason = parsed_content.get("reason")
                        agent_log.info(f"Agent reasoning: {reason}")
                except json.JSONDecodeError:
                    pass
            
            return content
            
        # 提取函数调用信息
        function_call = await extract_functions(content)
        
        if function_call:
            agent_log.debug(f"提取到函数调用: {function_call}")
            return content
        else:
            agent_log.debug(f"未提取到函数调用")
            return content
            
    except Exception as e:
        agent_log.error(f"处理OpenAI响应出错: {str(e)}")
        import traceback
        agent_log.error(f"错误详情: {traceback.format_exc()}")
        return f"Error processing OpenAI response: {str(e)}"

async def process_deepseek_response(response: Any) -> Dict[str, Any]:
    """处理DeepSeek模型的响应"""
    try:
        agent_log.debug(f"DeepSeek原始响应: {response}")
        
        # 提取content部分
        content = ""
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        else:
            if isinstance(response, tuple) and len(response) > 0:
                if hasattr(response[0], 'choices') and response[0].choices:
                    content = response[0].choices[0].message.content
                else:
                    content = str(response[0])
            else:
                content = str(response)
        
        agent_log.debug(f"提取的content: {content}")
        
        # 尝试解析JSON
        parsed_json = await parse_response_content(content)
        
        # 初始化返回结果
        result = {
            "action": {
                "name": "unknown_action",
                "arguments": {}
            },
            "reason": ""
        }
        
        if parsed_json:
            agent_log.debug(f"解析到有效JSON数据")
            
            # 提取reason
            if "reason" in parsed_json:
                result["reason"] = parsed_json["reason"]
            
            # 处理函数调用
            if "functions" in parsed_json and parsed_json["functions"]:
                first_function = parsed_json["functions"][0]
                result["action"] = {
                    "name": first_function.get("name", "unknown_action"),
                    "arguments": first_function.get("arguments", {})
                }
            elif "action_name" in parsed_json:
                result["action"] = {
                    "name": parsed_json.get("action_name", "unknown_action"),
                    "arguments": parsed_json.get("arguments", {})
                }
                
        return result
    
    except Exception as e:
        agent_log.error(f"DeepSeek响应处理失败: {str(e)}")
        import traceback
        agent_log.error(traceback.format_exc())
        return {
            "action": {
                "name": "unknown_action",
                "arguments": {}
            },
            "reason": "处理响应时发生错误"
        }

async def parse_response_content(content: str) -> Optional[Dict[str, Any]]:
    """解析响应内容中的JSON数据"""
    # 尝试多种方式提取和解析JSON
    parsed_json = None
    
    # 策略1: 提取Markdown代码块中的JSON
    json_block_patterns = [
        r'```json\s*(.*?)\s*```',  # 标准JSON代码块
        r'```\s*({\s*".*?})\s*```',  # 无标签代码块中的JSON
        r'```\s*(\[\s*{.*?}\s*\])\s*```'  # 无标签代码块中的JSON数组
    ]
    
    for pattern in json_block_patterns:
        try:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                json_str = match.group(1)
                parsed_json = json.loads(json_str)
                break
        except json.JSONDecodeError:
            continue
    
    # 策略2: 尝试直接解析整个content为JSON
    if parsed_json is None:
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # 策略3: 尝试在文本中查找JSON对象
    if parsed_json is None:
        json_patterns = [
            r'({[\s\S]*?"functions"[\s\S]*?})',  # 包含functions的JSON对象
            r'({[\s\S]*?"cognitive_state"[\s\S]*?})',  # 包含cognitive_state的JSON对象
            r'({[\s\S]*?"reason"[\s\S]*?})'  # 包含reason的JSON对象
        ]
        
        for pattern in json_patterns:
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    parsed_json = json.loads(json_str)
                    break
            except json.JSONDecodeError:
                continue
    
    return parsed_json

async def extract_functions(content: str) -> Optional[Dict[str, Any]]:
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