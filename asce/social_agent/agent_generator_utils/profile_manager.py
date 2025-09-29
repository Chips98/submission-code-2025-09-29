"""用户资料管理模块,用于处理代理的个人资料和认知状态。

包含个人资料创建、认知状态管理、用户信息对象构建等功能。
"""
import random
from typing import Any, Dict, List, Optional, Union

def create_user_profile(
    age: Optional[int] = None,
    gender: Optional[str] = None,
    country: Optional[str] = None,
    interests: Optional[List[str]] = None
) -> Dict[str, Any]:
    """创建用户个人资料"""
    profile = {
        "age": age or random.randint(18, 65),
        "gender": gender or random.choice(["male", "female"]),
        "country": country or "China",
        "interests": interests or []
    }
    return profile

def initialize_cognitive_state() -> Dict[str, Union[float, str]]:
    """初始化认知状态"""
    return {
        "mood": 0.0,
        "emotion": "neutral",
        "stance": "neutral",
        "thinking": "normal",
        "intention": "neutral"
    }

def build_user_info(
    profile: Dict[str, Any],
    cognitive_state: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """构建完整的用户信息对象"""
    user_info = {
        "profile": profile,
        "cognitive_state": cognitive_state or initialize_cognitive_state()
    }
    
    if additional_info:
        user_info.update(additional_info)
    
    return user_info

def generate_batch_profiles(
    names: List[str],
    mbti_types: List[str],
    ages: Optional[List[int]] = None,
    genders: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    interests_list: Optional[List[List[str]]] = None
) -> List[Dict[str, Any]]:
    """批量生成用户资料"""
    profiles = []
    for i, (name, mbti) in enumerate(zip(names, mbti_types)):
        profile = create_user_profile(
            age=ages[i] if ages else None,
            gender=genders[i] if genders else None,
            country=countries[i] if countries else None,
            interests=interests_list[i] if interests_list else None
        )
        profiles.append(profile)
    return profiles

def update_cognitive_state(
    current_state: Dict[str, Any],
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """更新认知状态"""
    new_state = current_state.copy()
    new_state.update(updates)
    return new_state

def validate_user_info(user_info: Dict[str, Any]) -> bool:
    """验证用户信息的完整性"""
    required_fields = {
        "profile": ["name", "mbti", "age", "gender", "country"],
        "cognitive_state": ["mood", "emotion", "stance", "thinking", "intention"]
    }
    
    try:
        for section, fields in required_fields.items():
            if section not in user_info:
                return False
            for field in fields:
                if field not in user_info[section]:
                    return False
        return True
    except Exception:
        return False 