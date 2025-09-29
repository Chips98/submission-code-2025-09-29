"""
ASCE社交代理响应解析模块

该模块提供了一组用于解析LLM响应的函数，专注于提取认知状态和函数调用信息。
设计目标是提高解析的稳定性和准确性，避免多种解析策略之间的冲突。
"""

import re
import json
import logging
import copy
from typing import Dict, List, Any, Tuple, Optional, Union

# 获取日志记录器
agent_log = logging.getLogger(name="social.agent")

class ResponseParser:
    """
    LLM响应解析器类

    提供一组方法用于从LLM响应中提取结构化数据，包括认知状态和函数调用信息。
    设计为与SocialAgent类配合使用，但保持相对独立以便于测试和维护。
    """

    def __init__(self, normalization_map=None, default_cognitive_profile=None, normalize_support_level_func=None, cognition_space_dict=None):
        """
        初始化响应解析器

        参数:
            normalization_map: 认知状态规范化映射
            default_cognitive_profile: 默认认知状态配置
            normalize_support_level_func: 支持级别规范化函数
            cognition_space_dict: 认知空间字典，用于获取有效的类型和值
        """
        self.normalization_map = normalization_map or {}

        # 默认认知状态配置
        self.default_cognitive_profile = default_cognitive_profile or {
            'mood': {'type': 'fail', 'value': 'fail'},
            'emotion': {'type': 'fail', 'value': 'fail'},
            'stance': {'type': 'fail', 'value': 'fail'},
            'cognition': {'type': 'fail', 'value': 'fail'},
            'intention': {'type': 'fail', 'value': 'fail'},
            'opinion': {
                'viewpoint_1': 'fail',
                'viewpoint_2': 'fail',
                'viewpoint_3': 'fail',
                'viewpoint_4': 'fail',
                'viewpoint_5': 'fail',
                'viewpoint_6': 'fail'
            }
        }

        # 支持级别规范化函数
        self.normalize_support_level_func = normalize_support_level_func

        # 认知空间字典
        self.cognition_space_dict = cognition_space_dict

    def extract_json_from_response(self, content: str) -> Tuple[Dict, str]:
        """
        从响应中提取JSON数据

        使用多种策略尝试提取有效的JSON数据，按优先级排序：
        1. 从Markdown代码块中提取JSON
        2. 尝试直接解析整个内容为JSON
        3. 在文本中查找JSON对象

        参数:
            content: LLM响应内容

        返回:
            Tuple[Dict, str]: 提取的JSON数据和使用的策略描述
        """
        # 策略1: 从Markdown代码块中提取JSON（使用原始内容）
        json_data, strategy = self._extract_json_from_code_blocks(content)
        if json_data:
            agent_log.info(f"成功从代码块提取JSON数据")
            return json_data, strategy

        # 预处理内容，修复常见格式问题
        content_fixed = self._preprocess_content(content)

        # 策略2: 尝试直接解析整个内容为JSON
        try:
            json_data = json.loads(content_fixed)
            agent_log.info(f"成功直接解析整个内容为JSON")
            return json_data, "直接解析整个内容为JSON"
        except json.JSONDecodeError as e:
            agent_log.debug(f"直接解析JSON失败: {e}")

        # 策略3: 在文本中查找JSON对象
        json_data, strategy = self._extract_json_from_text(content_fixed)
        if json_data:
            agent_log.info(f"成功从文本中提取JSON对象")
            return json_data, strategy

        # 所有策略都失败，返回空字典
        agent_log.warning(f"所有JSON提取策略均失败")
        return {}, "所有JSON提取策略均失败"

    def _preprocess_content(self, content: str) -> str:
        """
        预处理内容，修复常见格式问题

        参数:
            content: 原始内容

        返回:
            处理后的内容
        """
        try:
            # 替换双花括号为单花括号
            content_fixed = content.replace('{{', '{').replace('}}', '}')
            # 移除```json```标记
            content_fixed = re.sub(r'```(?:json)?|```', '', content_fixed).strip()
            # 修复常见格式问题
            content_fixed = re.sub(r',\s*}', '}', content_fixed)  # 移除JSON对象末尾多余的逗号
            content_fixed = re.sub(r',\s*]', ']', content_fixed)  # 移除数组末尾多余的逗号
            return content_fixed
        except Exception as e:
            agent_log.error(f"预处理内容时出错: {e}")
            return content  # 如果预处理失败，返回原始内容

    def _extract_json_from_code_blocks(self, content: str) -> Tuple[Dict, str]:
        """
        从代码块中提取JSON

        参数:
            content: 原始内容（未预处理）

        返回:
            Tuple[Dict, str]: 提取的JSON数据和策略描述
        """
        # 注意：这里使用原始内容，而不是预处理后的内容
        # 因为预处理会移除```标记

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
                    agent_log.debug(f"从代码块提取的JSON字符串: {json_str[:100]}..." if len(json_str) > 100 else f"从代码块提取的JSON字符串: {json_str}")
                    # 修复常见的JSON格式问题
                    json_str = re.sub(r',\s*}', '}', json_str)  # 移除对象末尾多余的逗号
                    json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号
                    return json.loads(json_str), "从代码块提取JSON"
            except json.JSONDecodeError as e:
                agent_log.debug(f"从代码块解析JSON失败: {e}")

        # 如果上面的模式匹配失败，尝试更简单的模式
        try:
            # 尝试匹配```和```之间的内容
            match = re.search(r'```(.*?)```', content, re.DOTALL)
            if match:
                # 提取代码块内容
                code_block = match.group(1).strip()
                # 如果代码块以json开头，移除这一行
                if code_block.startswith('json'):
                    code_block = code_block[4:].strip()

                # 修复常见的JSON格式问题
                code_block = re.sub(r',\s*}', '}', code_block)  # 移除对象末尾多余的逗号
                code_block = re.sub(r',\s*\]', ']', code_block)  # 移除数组末尾多余的逗号

                # 尝试解析为JSON
                agent_log.debug(f"尝试解析简化代码块: {code_block[:100]}..." if len(code_block) > 100 else f"尝试解析简化代码块: {code_block}")
                return json.loads(code_block), "从简化代码块提取JSON"
        except json.JSONDecodeError as e:
            agent_log.debug(f"从简化代码块解析JSON失败: {e}")
        except Exception as e:
            agent_log.debug(f"处理简化代码块时出错: {e}")

        # 如果上述方法都失败，尝试查找包含cognitive_state的JSON对象
        try:
            # 查找包含cognitive_state的JSON对象
            cognitive_match = re.search(r'\{[^\{\}]*"cognitive_state"[^\{\}]*\}', content, re.DOTALL)
            if cognitive_match:
                json_str = cognitive_match.group(0)
                # 修复常见的JSON格式问题
                json_str = re.sub(r',\s*}', '}', json_str)  # 移除对象末尾多余的逗号
                json_str = re.sub(r',\s*\]', ']', json_str)  # 移除数组末尾多余的逗号
                return json.loads(json_str), "从文本中提取包含cognitive_state的JSON对象"
        except json.JSONDecodeError as e:
            agent_log.debug(f"解析包含cognitive_state的JSON对象失败: {e}")
        except Exception as e:
            agent_log.debug(f"处理包含cognitive_state的JSON对象时出错: {e}")

        return {}, ""

    def _extract_json_from_text(self, content: str) -> Tuple[Dict, str]:
        """
        在文本中查找JSON对象

        参数:
            content: 预处理后的内容

        返回:
            Tuple[Dict, str]: 提取的JSON数据和策略描述
        """
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
                    agent_log.debug(f"从文本提取的JSON字符串: {json_str}")
                    return json.loads(json_str), "从文本中提取JSON对象"
            except json.JSONDecodeError as e:
                agent_log.debug(f"从文本解析JSON失败: {e}")

        return {}, ""

    def process_cognitive_state(self, json_data: Dict, current_cognitive_profile: Dict = None) -> Dict:
        """
        处理认知状态数据

        从JSON数据中提取认知状态，并与当前认知状态合并

        参数:
            json_data: 提取的JSON数据
            current_cognitive_profile: 当前认知状态

        返回:
            处理后的认知状态
        """
        # 如果没有提供当前认知状态，使用默认配置
        if current_cognitive_profile is None:
            current_cognitive_profile = copy.deepcopy(self.default_cognitive_profile)

        # 创建认知状态的备份，用于在提取失败时恢复
        cognitive_profile_backup = copy.deepcopy(current_cognitive_profile)

        # 从JSON数据中提取认知状态
        cognitive_data = self._extract_cognitive_data_from_json(json_data)
        if not cognitive_data:
            agent_log.warning("未能从JSON数据中提取认知状态")
            return current_cognitive_profile

        # 处理基本认知字段
        updated_profile = self._process_basic_cognitive_fields(cognitive_data, current_cognitive_profile)

        # 处理观点数据
        if "opinion" in cognitive_data:
            updated_profile = self._process_opinion_data(cognitive_data["opinion"], updated_profile)

        # 验证处理后的认知状态
        validated_profile = self._validate_cognitive_profile(updated_profile, cognitive_profile_backup)

        return validated_profile

    def _extract_cognitive_data_from_json(self, json_data: Dict) -> Dict:
        """
        从JSON数据中提取认知状态数据

        参数:
            json_data: 提取的JSON数据

        返回:
            认知状态数据
        """
        # 检查是否有cognitive_state子对象
        if "cognitive_state" in json_data and isinstance(json_data["cognitive_state"], dict):
            return json_data["cognitive_state"]
        # 检查是否有cognitive_profile子对象
        elif "cognitive_profile" in json_data and isinstance(json_data["cognitive_profile"], dict):
            return json_data["cognitive_profile"]
        # 如果没有专门的认知状态对象，尝试从顶层对象提取
        else:
            # 检查顶层对象是否包含认知字段
            cognitive_fields = ["mood", "emotion", "stance", "cognition", "intention", "opinion"]
            if any(field in json_data for field in cognitive_fields):
                return json_data

        return {}

    def _process_basic_cognitive_fields(self, cognitive_data: Dict, current_profile: Dict) -> Dict:
        """
        处理基本认知字段

        参数:
            cognitive_data: 认知状态数据
            current_profile: 当前认知状态

        返回:
            更新后的认知状态
        """
        # 创建一个新的认知状态对象，避免修改原始对象
        updated_profile = copy.deepcopy(current_profile)

        # 处理基本认知字段
        for field in ["mood", "emotion", "stance", "cognition", "intention"]:
            # 字段存在且为字典格式
            if field in cognitive_data and isinstance(cognitive_data[field], dict):
                field_data = cognitive_data[field]

                # 更新类型
                if "type" in field_data and field_data["type"]:
                    # 规范化类型
                    normalized_type = self._normalize_cognitive_type(field, field_data["type"])
                    updated_profile[field]["type"] = normalized_type

                # 更新值
                if "value" in field_data and field_data["value"]:
                    # 规范化值
                    normalized_value = self._normalize_cognitive_value(
                        field,
                        updated_profile[field]["type"],
                        field_data["value"]
                    )
                    updated_profile[field]["value"] = normalized_value

            # 字段为简单字符串格式
            elif field in cognitive_data and isinstance(cognitive_data[field], str):
                # 规范化类型
                normalized_type = self._normalize_cognitive_type(field, cognitive_data[field])
                updated_profile[field]["type"] = normalized_type

                # 规范化值
                normalized_value = self._normalize_cognitive_value(
                    field,
                    normalized_type,
                    cognitive_data[field]
                )
                updated_profile[field]["value"] = normalized_value

        return updated_profile

    def _process_opinion_data(self, opinion_data: Any, current_profile: Dict) -> Dict:
        """
        处理观点数据

        参数:
            opinion_data: 观点数据，可能是字典、列表或其他格式
            current_profile: 当前认知状态

        返回:
            更新后的认知状态
        """
        # 创建一个新的认知状态对象，避免修改原始对象
        updated_profile = copy.deepcopy(current_profile)

        # 确保opinion字段存在
        if "opinion" not in updated_profile:
            updated_profile["opinion"] = {}

        # 处理不同格式的opinion数据
        if isinstance(opinion_data, list):
            # 处理列表格式的opinion
            self._process_list_opinion(opinion_data, updated_profile)
        elif isinstance(opinion_data, dict):
            # 处理字典格式的opinion
            self._process_dict_opinion(opinion_data, updated_profile)

        # 确保所有viewpoint键都存在
        self._ensure_all_viewpoints_exist(updated_profile)

        return updated_profile

    def _process_list_opinion(self, opinion_list: List, profile: Dict) -> None:
        """
        处理列表格式的观点数据

        参数:
            opinion_list: 观点数据列表
            profile: 认知状态
        """
        for item in opinion_list:
            if not isinstance(item, dict):
                continue

            # 处理三种可能的格式:
            # 1. {"viewpoint_1": "支持级别"}
            # 2. {"viewpoint_n": "viewpoint_1", "type_support_levels": "支持级别"}
            # 3. {"viewpoint_1": "观点内容", "type_support_levels": "支持级别"}

            # 格式1: 直接包含viewpoint_n键且值为支持级别
            viewpoint_keys = [key for key in item if key.startswith("viewpoint_")]

            # 如果存在viewpoint_n键和type_support_levels键，则使用格式2或3
            if viewpoint_keys and "type_support_levels" in item:
                for key in viewpoint_keys:
                    support_level = item.get("type_support_levels")
                    if isinstance(support_level, str):
                        # 规范化支持级别
                        if self.normalize_support_level_func:
                            normalized_support = self.normalize_support_level_func(support_level)
                        else:
                            normalized_support = support_level
                        profile["opinion"][key] = normalized_support
                        agent_log.debug(f"从格式2/3提取观点: {key} = {normalized_support}")
            # 否则使用格式1
            else:
                for key in viewpoint_keys:
                    if isinstance(item[key], str):
                        # 规范化支持级别
                        if self.normalize_support_level_func:
                            normalized_support = self.normalize_support_level_func(item[key])
                        else:
                            normalized_support = item[key]
                        profile["opinion"][key] = normalized_support
                        agent_log.debug(f"从格式1提取观点: {key} = {normalized_support}")

            # 处理特殊格式: 包含viewpoint_n和type_support_levels，但viewpoint_n不是以viewpoint_开头
            if "viewpoint_n" in item and "type_support_levels" in item and not viewpoint_keys:
                viewpoint_key = item["viewpoint_n"]
                support_level = item["type_support_levels"]

                if isinstance(viewpoint_key, str) and isinstance(support_level, str):
                    # 如果viewpoint_key是数字，转换为viewpoint_N格式
                    if viewpoint_key.isdigit():
                        viewpoint_key = f"viewpoint_{viewpoint_key}"
                    # 如果viewpoint_key不是以viewpoint_开头，但是包含数字
                    elif not viewpoint_key.startswith("viewpoint_") and any(char.isdigit() for char in viewpoint_key):
                        # 提取数字部分
                        digit_part = ''.join(filter(str.isdigit, viewpoint_key))
                        if digit_part:
                            viewpoint_key = f"viewpoint_{digit_part}"

                    # 确保是有效的viewpoint键
                    if viewpoint_key.startswith("viewpoint_") and viewpoint_key[10:].isdigit():
                        # 规范化支持级别
                        if self.normalize_support_level_func:
                            normalized_support = self.normalize_support_level_func(support_level)
                        else:
                            normalized_support = support_level
                        profile["opinion"][viewpoint_key] = normalized_support
                        agent_log.debug(f"从特殊格式提取观点: {viewpoint_key} = {normalized_support}")

            # 处理新增格式: 当item中只有viewpoint_n和type_support_levels两个键
            # 这种情况在某些模型响应中很常见
            if len(item) == 2 and "type_support_levels" in item:
                # 找出另一个键，假设它是viewpoint键
                other_keys = [k for k in item.keys() if k != "type_support_levels"]
                if len(other_keys) == 1:
                    viewpoint_content = other_keys[0]
                    support_level = item["type_support_levels"]

                    # 尝试从viewpoint内容中提取viewpoint编号
                    if "viewpoint_" in viewpoint_content.lower():
                        # 尝试提取数字
                        digit_match = re.search(r'\d+', viewpoint_content)
                        if digit_match:
                            viewpoint_num = digit_match.group(0)
                            viewpoint_key = f"viewpoint_{viewpoint_num}"

                            # 规范化支持级别
                            if self.normalize_support_level_func:
                                normalized_support = self.normalize_support_level_func(support_level)
                            else:
                                normalized_support = support_level
                            profile["opinion"][viewpoint_key] = normalized_support
                            agent_log.debug(f"从新增格式提取观点: {viewpoint_key} = {normalized_support}")

    def _process_dict_opinion(self, opinion_dict: Dict, profile: Dict) -> None:
        """
        处理字典格式的观点数据

        参数:
            opinion_dict: 观点数据字典
            profile: 认知状态
        """
        # 直接格式: {"viewpoint_1": "support_level", ...}
        for key, value in opinion_dict.items():
            if key.startswith("viewpoint_") and isinstance(value, str):
                # 规范化支持级别
                if self.normalize_support_level_func:
                    normalized_support = self.normalize_support_level_func(value)
                else:
                    normalized_support = value
                profile["opinion"][key] = normalized_support

    def _ensure_all_viewpoints_exist(self, profile: Dict) -> None:
        """
        确保所有viewpoint键都存在

        参数:
            profile: 认知状态
        """
        # 确保opinion字段存在
        if "opinion" not in profile:
            profile["opinion"] = {}

        # 确保所有viewpoint键都存在
        for i in range(1, 7):
            key = f"viewpoint_{i}"
            if key not in profile["opinion"]:
                # 使用默认值
                default_value = self.default_cognitive_profile["opinion"].get(key, "Indifferent")
                profile["opinion"][key] = default_value

    def _validate_cognitive_profile(self, profile: Dict, backup_profile: Dict) -> Dict:
        """
        验证认知状态，确保所有字段都有有效值

        参数:
            profile: 处理后的认知状态
            backup_profile: 备份的认知状态

        返回:
            验证后的认知状态
        """
        # 检查基本认知字段
        for field in ["mood", "emotion", "stance", "cognition", "intention"]:
            if field not in profile or not isinstance(profile[field], dict):
                profile[field] = copy.deepcopy(backup_profile[field])
                continue

            # 检查type和value是否为fail或空
            field_data = profile[field]
            if "type" not in field_data or field_data["type"] in ["fail", ""]:
                field_data["type"] = backup_profile[field]["type"]

            if "value" not in field_data or field_data["value"] in ["fail", ""]:
                field_data["value"] = backup_profile[field]["value"]

        # 检查opinion字段
        if "opinion" not in profile or not isinstance(profile["opinion"], dict):
            profile["opinion"] = copy.deepcopy(backup_profile["opinion"])
        else:
            # 检查各个viewpoint是否为fail或空
            for i in range(1, 7):
                key = f"viewpoint_{i}"
                if key not in profile["opinion"] or profile["opinion"][key] in ["fail", ""]:
                    profile["opinion"][key] = backup_profile["opinion"].get(key, "Indifferent")

        return profile

    def extract_functions_from_response(self, content: str, json_data: Dict = None) -> List[Dict]:
        """
        从响应中提取函数调用信息

        参数:
            content: LLM响应内容
            json_data: 已提取的JSON数据，如果有的话

        返回:
            函数调用列表
        """
        # 如果提供了JSON数据，优先从中提取
        if json_data and "functions" in json_data and isinstance(json_data["functions"], list):
            return json_data["functions"]

        # 使用正则表达式提取函数调用
        functions = []

        # 使用更健壮的正则表达式匹配functions部分
        function_matches = re.finditer(r'"name"\s*:\s*"([^"]*)"[^{]*"arguments"\s*:\s*{([^}]*)}', content)

        for match in function_matches:
            name = match.group(1)
            args_str = match.group(2)

            arguments = {}
            # 改进参数匹配正则表达式，支持更复杂的值格式
            arg_matches = re.finditer(r'"([^"]*)"\s*:\s*(null|true|false|[0-9]+|"[^"]*")', args_str)

            for arg_match in arg_matches:
                arg_name = arg_match.group(1)
                arg_value = arg_match.group(2)

                # 处理不同类型的值
                if arg_value.startswith('"') and arg_value.endswith('"'):
                    arguments[arg_name] = arg_value[1:-1]  # 去除引号处理字符串
                elif arg_value == "null":
                    arguments[arg_name] = None
                elif arg_value == "true":
                    arguments[arg_name] = True
                elif arg_value == "false":
                    arguments[arg_name] = False
                else:
                    try:
                        arguments[arg_name] = int(arg_value)  # 处理整数
                    except ValueError:
                        try:
                            arguments[arg_name] = float(arg_value)  # 处理浮点数
                        except ValueError:
                            arguments[arg_name] = arg_value  # 保留原始值

            functions.append({
                "name": name,
                "arguments": arguments
            })

        # 如果上面的正则没有匹配到，尝试使用更宽松的模式
        if not functions:
            try:
                # 尝试直接从JSON字符串中提取functions数组
                functions_match = re.search(r'"functions"\s*:\s*(\[[\s\S]*?\])', content)
                if functions_match:
                    functions_str = functions_match.group(1)
                    functions = json.loads(functions_str)
                    agent_log.debug(f"通过直接JSON解析提取functions成功")
            except json.JSONDecodeError:
                agent_log.debug(f"直接JSON解析functions失败")

        return functions

    def extract_reason_from_response(self, content: str, json_data: Dict = None) -> str:
        """
        从响应中提取理由

        参数:
            content: LLM响应内容
            json_data: 已提取的JSON数据，如果有的话

        返回:
            提取的理由
        """
        # 如果提供了JSON数据，优先从中提取
        if json_data and "reason" in json_data:
            return json_data["reason"]

        # 使用正则表达式提取理由
        reason_patterns = [
            r'"reason"\s*:\s*"([^"]*)"',
            r'Reasoning: (.+?)(?=\n|$)',
            r'Reason: (.+?)(?=\n|$)',
            r'原因: (.+?)(?=\n|$)',
            r'意图: (.+?)(?=\n|$)'
        ]

        for pattern in reason_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                reason = match.group(1).strip()
                # 去除特殊字符
                reason = re.sub(r'[^\w\s,.?!;:\-\'"]', '', reason).strip()
                # 限制长度
                reason = reason[:100] + ("..." if len(reason) > 100 else "")
                return reason

        # 如果找不到特定模式，尝试直接提取语义上的理由
        lines = content.lower().split('\n')
        for i, line in enumerate(lines):
            if "reason" in line or "reasoning" in line or "原因" in line or "理由" in line or "意图" in line or "because" in line or "motivation" in line:
                if ":" in line:
                    return line.split(":", 1)[1].strip()[:100]
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()[:100]

        # 如果仍然没有理由，使用内容的前50个字符作为摘要
        if content:
            return content[:50] + ("..." if len(content) > 50 else "")

        return "No reason provided"

    def _normalize_cognitive_type(self, field: str, cog_type: str) -> str:
        """
        规范化认知类型

        参数:
            field: 认知维度字段名称
            cog_type: 原始值

        返回:
            规范化后的类型
        """
        # 默认类型
        default_type = 'neutral'
        if field == 'emotion':
            default_type = 'complex'  # emotion默认用complex
        elif field == 'cognition':
            default_type = 'analytical'  # cognition默认用analytical
        elif field == 'intention':
            default_type = 'expressive'  # intention默认用expressive

        # 如果类型为空，返回默认类型
        if not cog_type:
            return default_type

        # 将输入类型转换为小写用于匹配
        cog_type_lower = str(cog_type).lower()

        # 尝试从规范化映射中获取类型映射
        field_type_key = f"{field}_type"
        if self.normalization_map and field_type_key in self.normalization_map:
            # 遍历标准类型及其变体
            for standard_type, variants in self.normalization_map[field_type_key].items():
                # 精确匹配
                if cog_type_lower == standard_type.lower():
                    return standard_type

                # 变体匹配
                for variant in variants:
                    variant_lower = variant.lower()
                    # 完全匹配
                    if variant_lower == cog_type_lower:
                        return standard_type
                    # 部分匹配
                    if variant_lower in cog_type_lower or cog_type_lower in variant_lower:
                        return standard_type

        # 从认知空间字典获取有效类型列表
        if self.cognition_space_dict and field in self.cognition_space_dict and "type_list" in self.cognition_space_dict[field]:
            valid_types = self.cognition_space_dict[field]["type_list"]

            # 精确匹配
            for valid_type in valid_types:
                if valid_type.lower() == cog_type_lower:
                    return valid_type

            # 模糊匹配
            for valid_type in valid_types:
                if valid_type.lower() in cog_type_lower or cog_type_lower in valid_type.lower():
                    return valid_type

            # 如果没有匹配到，返回第一个有效类型
            if valid_types:
                return valid_types[0]

        # 如果没有认知空间字典或没有匹配到，返回默认类型
        return default_type

    def _normalize_cognitive_value(self, field: str, cog_type: str, cog_value: str) -> str:
        """
        规范化认知值

        参数:
            field: 认知维度字段名称
            cog_type: 规范化后的类型
            cog_value: 原始值

        返回:
            规范化后的值
        """
        # 如果值为空，使用类型作为值
        if not cog_value:
            return cog_type

        # 将输入值转换为字符串并小写用于匹配
        cog_value_lower = str(cog_value).lower()

        # 处理多值情况（以逗号、and、or或其他分隔符分隔的多个值）
        if ',' in cog_value_lower or ' and ' in cog_value_lower or '/' in cog_value_lower or ' or ' in cog_value_lower:
            # 提取可能的多个值
            value_parts = re.split(r',|\s+and\s+|/|\s+or\s+', cog_value_lower)
            value_parts = [part.strip() for part in value_parts if part.strip()]

            # 只保留第一个值
            if value_parts:
                cog_value_lower = value_parts[0]

        # 尝试从规范化映射中获取值映射
        field_value_key = f"{field}_value"
        if self.normalization_map and field_value_key in self.normalization_map:
            # 遍历标准值及其变体
            for standard_value, variants in self.normalization_map[field_value_key].items():
                # 精确匹配
                if cog_value_lower == standard_value.lower():
                    return standard_value

                # 变体匹配
                for variant in variants:
                    variant_lower = variant.lower()
                    # 完全匹配
                    if variant_lower == cog_value_lower:
                        return standard_value
                    # 部分匹配
                    if variant_lower in cog_value_lower or cog_value_lower in variant_lower:
                        return standard_value

        # 从认知空间字典获取有效值列表
        if self.cognition_space_dict and field in self.cognition_space_dict and "value_list" in self.cognition_space_dict[field]:
            value_list = self.cognition_space_dict[field]["value_list"]

            # 如果值列表是字典，根据类型获取对应的值列表
            if isinstance(value_list, dict) and cog_type in value_list:
                type_values = value_list[cog_type]

                # 精确匹配
                for valid_value in type_values:
                    if valid_value.lower() == cog_value_lower:
                        return valid_value

                # 模糊匹配
                for valid_value in type_values:
                    if valid_value.lower() in cog_value_lower or cog_value_lower in valid_value.lower():
                        return valid_value

                # 如果没有匹配到，返回第一个有效值
                if type_values:
                    return type_values[0]

            # 如果值列表是列表，直接使用
            elif isinstance(value_list, list):
                # 精确匹配
                for valid_value in value_list:
                    if valid_value.lower() == cog_value_lower:
                        return valid_value

                # 模糊匹配
                for valid_value in value_list:
                    if valid_value.lower() in cog_value_lower or cog_value_lower in valid_value.lower():
                        return valid_value

                # 如果没有匹配到，返回第一个有效值
                if value_list:
                    return value_list[0]

        # 如果没有匹配到有效值，返回原始值的首字母大写形式
        return cog_value[0].upper() + cog_value[1:] if cog_value else cog_type

    def extract_regex_cognitive_state(self, content: str) -> Dict:
        """
        使用正则表达式从响应中提取认知状态

        当JSON解析失败时的备用方法

        参数:
            content: LLM响应内容

        返回:
            提取的认知状态
        """
        cognitive_state = {}
        cognitive_fields = ["mood", "emotion", "stance", "cognition", "intention"]

        for field in cognitive_fields:
            # 更强大的正则模式 - 同时支持多种格式
            field_match = re.search(fr'"{field}"\s*:\s*(?:"([^"]*)"|{{\s*"type"\s*:\s*"([^"]*)"(?:,\s*"value"\s*:\s*"([^"]*)"|)}})', content, re.DOTALL)

            if field_match:
                if field_match.group(1):  # 简单字符串值
                    # 规范化类型和值
                    normalized_type = self._normalize_cognitive_type(field, field_match.group(1))
                    normalized_value = self._normalize_cognitive_value(field, normalized_type, field_match.group(1))
                    cognitive_state[field] = {
                        "type": normalized_type,
                        "value": normalized_value
                    }
                elif field_match.group(2):  # 对象格式
                    # 规范化类型
                    normalized_type = self._normalize_cognitive_type(field, field_match.group(2))
                    # 如果有value字段，使用它，否则使用type作为value的输入
                    value_input = field_match.group(3) if field_match.group(3) else field_match.group(2)
                    normalized_value = self._normalize_cognitive_value(field, normalized_type, value_input)
                    cognitive_state[field] = {
                        "type": normalized_type,
                        "value": normalized_value
                    }

        # 提取opinion信息
        opinion = {}

        # 方法1: 查找包含viewpoint_1到viewpoint_6的部分，格式为: "viewpoint_1": "支持级别"
        viewpoint_matches = re.finditer(r'"(viewpoint_[1-6])"\s*:\s*"([^"]*)"', content)
        for match in viewpoint_matches:
            viewpoint_key = match.group(1)
            support_level = match.group(2)
            if self.normalize_support_level_func:
                normalized_support = self.normalize_support_level_func(support_level)
            else:
                normalized_support = support_level
            opinion[viewpoint_key] = normalized_support

        # 方法2: 查找数组形式的观点, 格式为: { "viewpoint_1": "支持级别" }
        array_opinion_matches = re.finditer(r'{\s*"(viewpoint_[1-6])"\s*:\s*"([^"]*)"\s*}', content)
        for match in array_opinion_matches:
            viewpoint_key = match.group(1)
            support_level = match.group(2)
            if self.normalize_support_level_func:
                normalized_support = self.normalize_support_level_func(support_level)
            else:
                normalized_support = support_level
            opinion[viewpoint_key] = normalized_support

        # 方法3: 查找带有type_support_levels的观点格式
        # 格式为: {"viewpoint_1": "...", "type_support_levels": "支持级别"}
        type_support_matches = re.finditer(r'\{\s*"(viewpoint_[1-6])"\s*:\s*"[^"]*"\s*,\s*"type_support_levels"\s*:\s*"([^"]*)"\s*\}', content)
        for match in type_support_matches:
            viewpoint_key = match.group(1)
            support_level = match.group(2)
            if self.normalize_support_level_func:
                normalized_support = self.normalize_support_level_func(support_level)
            else:
                normalized_support = support_level
            opinion[viewpoint_key] = normalized_support

        # 如果提取到观点数据，添加到认知状态
        if opinion:
            cognitive_state["opinion"] = opinion
        else:
            # 如果没有提取到观点数据，尝试使用更宽松的正则表达式
            # 查找opinion数组
            opinion_array_match = re.search(r'"opinion"\s*:\s*(\[.*?\])', content, re.DOTALL)
            if opinion_array_match:
                try:
                    # 尝试解析opinion数组
                    opinion_array_str = opinion_array_match.group(1)
                    # 修复常见的JSON格式问题
                    opinion_array_str = re.sub(r',\s*\]', ']', opinion_array_str)  # 移除数组末尾多余的逗号
                    opinion_array = json.loads(opinion_array_str)

                    # 处理opinion数组
                    for item in opinion_array:
                        if isinstance(item, dict):
                            # 查找viewpoint键
                            viewpoint_keys = [k for k in item.keys() if k.startswith("viewpoint_")]
                            if viewpoint_keys and "type_support_levels" in item:
                                for key in viewpoint_keys:
                                    support_level = item["type_support_levels"]
                                    if isinstance(support_level, str):
                                        if self.normalize_support_level_func:
                                            normalized_support = self.normalize_support_level_func(support_level)
                                        else:
                                            normalized_support = support_level
                                        opinion[key] = normalized_support

                    if opinion:
                        cognitive_state["opinion"] = opinion
                except Exception as e:
                    agent_log.debug(f"解析opinion数组失败: {e}")

        return cognitive_state
