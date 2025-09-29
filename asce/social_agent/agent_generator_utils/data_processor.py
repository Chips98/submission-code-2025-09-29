"""数据处理模块,用于处理代理生成过程中的各类数据。

包含数据格式转换、批量处理、数据验证等功能。
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

def convert_to_json(data: Any) -> str:
    """将数据转换为JSON字符串"""
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"JSON转换失败: {str(e)}")

def parse_json(json_str: str) -> Any:
    """解析JSON字符串"""
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON解析失败: {str(e)}")

def batch_process(
    data_list: List[Any],
    process_func: callable,
    batch_size: int = 100
) -> List[Any]:
    """批量处理数据"""
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
    return results

def validate_data_format(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """验证数据格式"""
    errors = []
    
    def _validate_field(value: Any, field_schema: Dict[str, Any], path: str) -> None:
        field_type = field_schema.get("type")
        required = field_schema.get("required", False)
        
        if required and value is None:
            errors.append(f"{path} 是必需的")
            return
            
        if value is not None:
            if field_type == "string" and not isinstance(value, str):
                errors.append(f"{path} 必须是字符串")
            elif field_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{path} 必须是数字")
            elif field_type == "array" and not isinstance(value, list):
                errors.append(f"{path} 必须是数组")
            elif field_type == "object" and not isinstance(value, dict):
                errors.append(f"{path} 必须是对象")
    
    for field_name, field_schema in schema.items():
        value = data.get(field_name)
        _validate_field(value, field_schema, field_name)
    
    return len(errors) == 0, errors

def clean_data(
    data: Dict[str, Any],
    rules: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """清理数据"""
    cleaned = {}
    
    for field, value in data.items():
        if field not in rules:
            cleaned[field] = value
            continue
            
        rule = rules[field]
        field_type = rule.get("type")
        
        if value is None:
            cleaned[field] = rule.get("default")
            continue
            
        try:
            if field_type == "string":
                cleaned[field] = str(value).strip()
            elif field_type == "number":
                cleaned[field] = float(value)
            elif field_type == "integer":
                cleaned[field] = int(value)
            elif field_type == "boolean":
                cleaned[field] = bool(value)
            elif field_type == "array":
                if isinstance(value, list):
                    cleaned[field] = value
                else:
                    cleaned[field] = [value]
            else:
                cleaned[field] = value
        except Exception:
            cleaned[field] = rule.get("default")
            
    return cleaned

def merge_data(
    data_list: List[Dict[str, Any]],
    merge_strategy: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """合并多个数据字典"""
    if not data_list:
        return {}
        
    result = {}
    merge_strategy = merge_strategy or {}
    
    for data in data_list:
        for key, value in data.items():
            strategy = merge_strategy.get(key, "override")
            
            if key not in result:
                result[key] = value
            elif strategy == "append" and isinstance(result[key], list):
                if isinstance(value, list):
                    result[key].extend(value)
                else:
                    result[key].append(value)
            elif strategy == "sum" and isinstance(result[key], (int, float)):
                result[key] += value
            elif strategy == "override":
                result[key] = value
                
    return result

def extract_fields(
    data: Dict[str, Any],
    field_paths: List[str],
    default_value: Any = None
) -> Dict[str, Any]:
    """提取指定字段"""
    result = {}
    
    for path in field_paths:
        keys = path.split(".")
        value = data
        try:
            for key in keys:
                value = value[key]
            result[path] = value
        except (KeyError, TypeError):
            result[path] = default_value
            
    return result 