"""配置处理模块,用于处理代理生成相关的各种配置。

包含模型配置处理、随机种子设置、MBTI类型管理等功能。
"""
import random
from typing import Any, List, Dict, Tuple

from camel.types import ModelType

def setup_model_configs(
    cfgs: List[Dict[str, Any]], 
    num_agents: int
) -> Tuple[List[ModelType], List[float], Dict[ModelType, Dict[str, Any]]]:
    """设置模型配置,返回模型类型列表、温度列表和配置字典"""
    model_types = []
    model_temperatures = []
    model_config_dict = {}
    
    # 如果cfgs为None,使用默认配置
    if cfgs is None:
        default_cfg = {
            "model_type": "gpt-3.5-turbo",
            "num": num_agents,
            "temperature": 0.7
        }
        cfgs = [default_cfg]
    
    for cfg in cfgs:
        model_type = ModelType(cfg["model_type"])
        model_config_dict[model_type] = cfg
        model_types.extend([model_type] * cfg["num"])
        temperature = cfg.get("temperature", 0.0)
        model_temperatures.extend([temperature] * cfg["num"])
        
    random.shuffle(model_types)
    assert len(model_types) == num_agents
    
    return model_types, model_temperatures, model_config_dict

def get_mbti_types() -> List[str]:
    """获取MBTI人格类型列表"""
    return ["INTJ", "ENTP", "INFJ", "ENFP"]

def process_activity_frequency(frequency_list: List[str]) -> List[float]:
    """处理活动频率列表,返回概率列表"""
    import ast
    import numpy as np
    
    all_freq = np.array([ast.literal_eval(fre) for fre in frequency_list])
    normalized_prob = np.ones_like(all_freq)
    return normalized_prob.tolist()

def validate_model_count(model_types: List[ModelType], agent_count: int) -> None:
    """验证模型数量与代理数量是否匹配"""
    assert len(model_types) == agent_count, (
        f"模型数量({len(model_types)})与代理数量({agent_count})不匹配"
    ) 