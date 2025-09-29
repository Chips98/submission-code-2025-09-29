"""动作执行模块

此模块包含执行代理动作的相关函数。
"""

import logging
import json
import re
from typing import Dict, Any, Optional

# 配置日志
agent_log = logging.getLogger(name="social.agent")

async def process_function_call(function_call: Dict[str, Any], env_action: Any) -> Dict[str, Any]:
    """处理函数调用"""
    try:
        function_name = function_call.get("name", "")
        arguments = function_call.get("arguments", {})
        
        # 匹配真实动作
        matching_action = None
        for func_name in dir(env_action):
            if func_name.startswith('_'): 
                continue
            
            func = getattr(env_action, func_name)  
            
            if callable(func) and not func_name.startswith('_'):
                if func_name == function_name:
                    matching_action = func_name
                    break
        
        if matching_action:
            result = await getattr(env_action, matching_action)(**arguments)
            return result
        else:
            print(f"未知函数: {function_name}")
            return {"success": False, "error": f"Unknown function: {function_name}"}
        
    except Exception as e:
        agent_log.error(f"处理函数调用出错: {str(e)}")
        import traceback
        agent_log.error(f"错误详情: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}

def perform_agent_graph_action(agent_id: int, action_name: str, arguments: Dict[str, Any], agent_graph: Any) -> None:
    """根据动作类型修改代理图结构"""
    # 获取被关注者ID
    followee_id = arguments.get("followee_id")
    if followee_id is None:
        return
        
    # 根据动作类型执行相应操作
    if "unfollow" in action_name:
        agent_graph.remove_edge(agent_id, followee_id)
        agent_log.debug(f"Agent {agent_id} unfollowed {followee_id}")
    elif "follow" in action_name:
        agent_graph.add_edge(agent_id, followee_id)
        agent_log.debug(f"Agent {agent_id} followed {followee_id}")

async def handle_function_calls(functions: list, content: str, env_action: Any) -> str:
    """统一处理函数调用逻辑"""
    try:
        func_info = functions[0]
        called_name = func_info.get("name", "") or func_info.get("action", "")
        args = func_info.get("arguments", {}) or func_info.get("params", {})
        
        # 执行动作
        result = await process_function_call({"name": called_name, "arguments": args}, env_action)
        return content[:1000]
    except Exception as e:
        agent_log.error(f"函数调用处理失败: {str(e)}")
        return content 