"""
ASCE Social Agent Camel Response Parser Module

This module provides functionality to parse LLM responses, focusing on extracting cognitive states
and function calls. It's designed to be robust and work even when external dependencies are not available.
"""

import logging
import json
import re
import copy
from typing import Dict, List, Any, Tuple, Optional, Union

# Set up logging
agent_log = logging.getLogger(name="social.agent")

# Check if Camel framework is available
CAMEL_AVAILABLE = False
OUTLINES_AVAILABLE = False

# Try to import from camel.types (newer versions)
try:
    from camel.types import OutlinesConverter
    CAMEL_AVAILABLE = True
    OUTLINES_AVAILABLE = True
    agent_log.info("Successfully imported OutlinesConverter from camel.types")
except ImportError:
    # Try to import directly from outlines (if installed separately)
    try:
        import outlines
        OUTLINES_AVAILABLE = True
        agent_log.info("Successfully imported outlines module")
    except ImportError:
        agent_log.warning("Neither camel.types nor outlines module is available. Will use fallback parsing method.")

# Import custom schemas if available
try:
    from asce.social_agent.camel_schemas import ASCEResponse, CognitiveState, Opinion, Function
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    agent_log.warning("Custom schemas not available. Will use dictionary-based validation.")


class CamelResponseParser:
    """
    使用Camel框架的LLM响应解析器类

    提供一组方法用于从LLM响应中提取结构化数据，包括认知状态和函数调用信息。
    使用Camel的OutlinesConverter进行结构化解析，提高解析的稳定性。
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
            'thinking': {'type': 'fail', 'value': 'fail'},
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

        # 检查Camel是否可用
        self.camel_available = CAMEL_AVAILABLE
        if not self.camel_available:
            agent_log.warning("Camel framework not available, cannot use Camel parsing functionality")

        # 初始化Camel转换器
        self._init_camel_converter()

    def _init_camel_converter(self):
        """Initialize Camel converter"""
        # Check if Camel or Outlines is available
        if not CAMEL_AVAILABLE and not OUTLINES_AVAILABLE:
            self.converter = None
            self.camel_available = False
            agent_log.warning("Neither Camel nor Outlines is available, cannot initialize converter")
            return

        try:
            # If Camel is available, use its OutlinesConverter
            if CAMEL_AVAILABLE:
                self.converter = OutlinesConverter(
                    model_type="llama3",  # Using a generic model type
                    platform="vllm"
                )
                agent_log.info("Successfully initialized Camel OutlinesConverter")
            # If only Outlines is available, use it directly
            elif OUTLINES_AVAILABLE:
                try:
                    # Create a simple wrapper that mimics OutlinesConverter
                    class OutlinesWrapper:
                        def convert(self, content, type, output_schema):
                            # Use outlines to parse the content
                            try:
                                from outlines import parse
                                return parse(content, output_schema)
                            except Exception as parse_error:
                                agent_log.error(f"Failed to parse with outlines: {parse_error}")
                                raise

                    self.converter = OutlinesWrapper()
                    agent_log.info("Successfully initialized Outlines wrapper")
                except Exception as wrapper_error:
                    agent_log.error(f"Failed to initialize Outlines wrapper: {wrapper_error}")
                    raise

            self.camel_available = True
        except Exception as e:
            agent_log.error(f"Failed to initialize converter: {e}")
            self.converter = None
            self.camel_available = False

    def parse_response(self, content: str) -> Tuple[Dict, str]:
        """
        Parse LLM response

        Uses multiple strategies to extract structured data from LLM responses:
        1. Extract from <asce_response> tags
        2. Use Camel/Outlines converter if available
        3. Extract from code blocks
        4. Extract from regular JSON patterns

        Args:
            content: LLM response content

        Returns:
            Tuple[Dict, str]: Parsed data and description of the strategy used
        """
        # Strategy 1: Extract from <asce_response> tags
        try:
            # Look for <asce_response> tags
            import re
            pattern = r'<asce_response>(.*?)</asce_response>'
            match = re.search(pattern, content, re.DOTALL)

            if match:
                # Extract the content between tags
                json_str = match.group(1).strip()
                try:
                    # Try to parse as JSON
                    json_data = json.loads(json_str)
                    # Try to validate with Pydantic model if available
                    try:
                        parsed_data = ASCEResponse(**json_data)
                        # Convert to dict
                        result = parsed_data.model_dump() if hasattr(parsed_data, 'model_dump') else parsed_data.dict()
                        agent_log.info("Successfully parsed response using tag extraction with schema validation")
                        return result, "Tag extraction with schema validation"
                    except Exception as schema_error:
                        # If schema validation fails, use the raw JSON
                        agent_log.warning(f"Schema validation failed: {schema_error}, using raw JSON")
                        agent_log.info("Successfully parsed response using tag extraction (raw JSON)")
                        return json_data, "Tag extraction (raw JSON)"
                except json.JSONDecodeError as json_error:
                    agent_log.warning(f"Failed to parse JSON from tags: {json_error}")
            else:
                agent_log.debug("No <asce_response> tags found in content")
        except Exception as e:
            agent_log.warning(f"Error in tag extraction: {e}")

        # Strategy 2: Use Camel/Outlines converter if available
        if self.camel_available and self.converter is not None:
            try:
                # Use the converter to parse the response
                parsed_data = self.converter.convert(
                    content=content,
                    type="json",
                    output_schema=ASCEResponse
                )

                # Convert Pydantic model to dict
                result = parsed_data.model_dump() if hasattr(parsed_data, 'model_dump') else parsed_data.dict()
                agent_log.info("Successfully parsed response using converter")
                return result, "Converter parsing"
            except Exception as e:
                agent_log.warning(f"Failed to parse response using converter: {e}")
                # Continue to next strategy
        else:
            agent_log.info("Converter not available, skipping converter parsing strategy")

        # Strategy 3: Extract from code blocks
        try:
            # Look for JSON in code blocks
            pattern = r'```(?:json)?\s*([\s\S]*?)```'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                try:
                    # Try to parse as JSON
                    json_str = match.strip()
                    json_data = json.loads(json_str)

                    # Check if it has the expected structure (cognitive_state and functions)
                    if isinstance(json_data, dict) and "cognitive_state" in json_data:
                        agent_log.info("Successfully parsed response from code block")
                        return json_data, "Code block extraction"
                except json.JSONDecodeError:
                    continue

            agent_log.debug("No valid JSON found in code blocks")
        except Exception as e:
            agent_log.warning(f"Error in code block extraction: {e}")

        # Strategy 4: Look for any JSON object in the content
        try:
            # Find all JSON-like patterns
            pattern = r'\{[^\{\}]*\"cognitive_state\"[^\{\}]*\}'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                try:
                    json_data = json.loads(match)
                    if isinstance(json_data, dict) and "cognitive_state" in json_data:
                        agent_log.info("Successfully parsed response using JSON pattern extraction")
                        return json_data, "JSON pattern extraction"
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            agent_log.warning(f"Error in JSON pattern extraction: {e}")

        # If all strategies fail, return empty dict
        agent_log.error("All parsing strategies failed")
        return {}, "All parsing strategies failed"

    def extract_cognitive_state(self, json_data: Dict) -> Dict:
        """
        Extract cognitive state from parsed JSON data

        Args:
            json_data: Parsed JSON data

        Returns:
            Dict: Extracted cognitive state
        """
        # If JSON data is empty, return default cognitive profile
        if not json_data:
            agent_log.warning("Empty JSON data, using default cognitive profile")
            return copy.deepcopy(self.default_cognitive_profile)

        # Create a copy of the default cognitive profile to use as a base
        cognitive_profile = copy.deepcopy(self.default_cognitive_profile)

        try:
            # Extract cognitive state
            if "cognitive_state" in json_data:
                cog_state = json_data["cognitive_state"]
                if not isinstance(cog_state, dict):
                    agent_log.warning(f"cognitive_state is not a dictionary: {type(cog_state)}, using default")
                    return cognitive_profile

                # Process each cognitive dimension
                for field in ["mood", "emotion", "stance", "thinking", "intention"]:
                    try:
                        if field in cog_state:
                            field_data = cog_state[field]

                            # Handle different formats
                            if isinstance(field_data, dict) and "type" in field_data and "value" in field_data:
                                # Standard format: {"type": "...", "value": "..."}
                                cog_type = str(field_data.get("type", ""))
                                cog_value = str(field_data.get("value", ""))

                                # Try to normalize if possible
                                try:
                                    if hasattr(self, "_normalize_cognitive_type"):
                                        cog_type = self._normalize_cognitive_type(field, cog_type)
                                    if hasattr(self, "_normalize_cognitive_value"):
                                        cog_value = self._normalize_cognitive_value(field, cog_type, cog_value)
                                except Exception as norm_error:
                                    agent_log.warning(f"Failed to normalize {field}: {norm_error}")

                                cognitive_profile[field] = {
                                    "type": cog_type,
                                    "value": cog_value
                                }
                            elif isinstance(field_data, str):
                                # Simple string format
                                cognitive_profile[field] = {
                                    "type": field,  # Use field name as type
                                    "value": str(field_data)
                                }
                    except Exception as field_error:
                        agent_log.warning(f"Error processing {field}: {field_error}")

            # Process opinion data
            if "opinion" in json_data:
                opinion_data = json_data["opinion"]

                # Ensure opinion field exists
                if "opinion" not in cognitive_profile:
                    cognitive_profile["opinion"] = {}

                # Handle different opinion formats
                if isinstance(opinion_data, dict):
                    # Standard dictionary format
                    for i in range(1, 7):
                        key = f"viewpoint_{i}"
                        if key in opinion_data:
                            value = opinion_data[key]
                            # Normalize support level if function is available
                            if self.normalize_support_level_func:
                                try:
                                    normalized_support = self.normalize_support_level_func(value)
                                    cognitive_profile["opinion"][key] = normalized_support
                                except Exception as norm_error:
                                    agent_log.warning(f"Failed to normalize support level for {key}: {norm_error}")
                                    cognitive_profile["opinion"][key] = str(value)
                            else:
                                cognitive_profile["opinion"][key] = str(value)
                elif isinstance(opinion_data, list):
                    # List format (old format or alternative format)
                    for item in opinion_data:
                        if isinstance(item, dict):
                            # Try to extract viewpoint and support level
                            for key, value in item.items():
                                if key.startswith("viewpoint_"):
                                    support_level = item.get("type_support_levels", "")
                                    if self.normalize_support_level_func:
                                        try:
                                            support_level = self.normalize_support_level_func(support_level)
                                        except Exception:
                                            pass
                                    cognitive_profile["opinion"][key] = support_level

            return cognitive_profile
        except Exception as e:
            agent_log.error(f"Failed to extract cognitive state: {e}")
            return copy.deepcopy(self.default_cognitive_profile)

    def extract_functions(self, json_data: Dict) -> List[Dict]:
        """
        Extract function calls from parsed JSON data

        Args:
            json_data: Parsed JSON data

        Returns:
            List[Dict]: Extracted function calls
        """
        # If JSON data is empty, return empty list
        if not json_data:
            return []

        try:
            functions = []

            # Strategy 1: Extract from 'functions' list
            if "functions" in json_data:
                funcs_data = json_data["functions"]

                # Handle list format
                if isinstance(funcs_data, list):
                    for func in funcs_data:
                        if isinstance(func, dict) and "name" in func:
                            # Extract function name and arguments
                            name = func.get("name", "")
                            arguments = func.get("arguments", {})

                            # Ensure arguments is a dictionary
                            if not isinstance(arguments, dict):
                                agent_log.warning(f"Function arguments is not a dictionary: {type(arguments)}, using empty dict")
                                arguments = {}

                            # Add to function calls list
                            functions.append({
                                "name": name,
                                "arguments": arguments
                            })
                # Handle single function object
                elif isinstance(funcs_data, dict) and "name" in funcs_data:
                    name = funcs_data.get("name", "")
                    arguments = funcs_data.get("arguments", {})

                    # Ensure arguments is a dictionary
                    if not isinstance(arguments, dict):
                        arguments = {}

                    functions.append({
                        "name": name,
                        "arguments": arguments
                    })

            # Strategy 2: Extract from 'function' object (singular)
            if not functions and "function" in json_data:
                func = json_data["function"]
                if isinstance(func, dict) and "name" in func:
                    name = func.get("name", "")
                    arguments = func.get("arguments", {})

                    # Ensure arguments is a dictionary
                    if not isinstance(arguments, dict):
                        arguments = {}

                    functions.append({
                        "name": name,
                        "arguments": arguments
                    })

            # Strategy 3: Look for function-like objects at the root level
            if not functions:
                for key, value in json_data.items():
                    if key.startswith("function_") and isinstance(value, dict):
                        if "name" in value:
                            name = value.get("name", "")
                            arguments = value.get("arguments", {})

                            # Ensure arguments is a dictionary
                            if not isinstance(arguments, dict):
                                arguments = {}

                            functions.append({
                                "name": name,
                                "arguments": arguments
                            })

            return functions
        except Exception as e:
            agent_log.error(f"Failed to extract function calls: {e}")
            return []

    def extract_reason(self, json_data: Dict) -> str:
        """
        Extract reasoning from parsed JSON data

        Args:
            json_data: Parsed JSON data

        Returns:
            str: Extracted reasoning
        """
        # If JSON data is empty, return default reason
        if not json_data:
            return "No reason provided"

        try:
            # Strategy 1: Extract from 'reason' field
            if "reason" in json_data:
                if isinstance(json_data["reason"], str):
                    reason = json_data["reason"].strip()
                    # Limit length
                    reason = reason[:500] + ("..." if len(reason) > 500 else "")
                    return reason
                else:
                    agent_log.warning(f"Reason is not a string: {type(json_data['reason'])}")

            # Strategy 2: Extract from 'reasoning' field
            if "reasoning" in json_data and isinstance(json_data["reasoning"], str):
                reason = json_data["reasoning"].strip()
                # Limit length
                reason = reason[:500] + ("..." if len(reason) > 500 else "")
                return reason

            # Strategy 3: Extract from 'thought' field
            if "thought" in json_data and isinstance(json_data["thought"], str):
                reason = json_data["thought"].strip()
                # Limit length
                reason = reason[:500] + ("..." if len(reason) > 500 else "")
                return reason

            return "No reason provided"
        except Exception as e:
            agent_log.error(f"Failed to extract reason: {e}")
            return "No reason provided"

    def _normalize_cognitive_type(self, field: str, cog_type: str) -> str:
        """
        Normalize cognitive type

        Args:
            field: Cognitive dimension field name
            cog_type: Original type value

        Returns:
            Normalized type
        """
        # Default types for different fields
        default_types = {
            'mood': 'neutral',
            'emotion': 'complex',
            'stance': 'neutral',
            'thinking': 'analytical',
            'intention': 'expressive'
        }

        # Get default type for this field
        default_type = default_types.get(field, 'neutral')

        # If type is empty or None, return default
        if not cog_type:
            return default_type

        # Convert to string and lowercase for comparison
        try:
            cog_type_str = str(cog_type).strip()
            cog_type_lower = cog_type_str.lower()
        except Exception:
            return default_type

        # If empty after conversion, return default
        if not cog_type_lower:
            return default_type

        # If we have a normalization map, try to use it
        if self.normalization_map and field in self.normalization_map:
            field_map = self.normalization_map.get(field, {})

            # Check for exact match in normalization map
            if cog_type_lower in field_map:
                return field_map[cog_type_lower]

            # Check for partial match
            for norm_key, norm_value in field_map.items():
                if norm_key in cog_type_lower or cog_type_lower in norm_key:
                    return norm_value

        # If we have a cognition space dictionary, try to use it
        if self.cognition_space_dict and field in self.cognition_space_dict:
            # Get valid types list
            valid_types = self.cognition_space_dict.get(field, {}).get('type_list', [])
            if not valid_types:
                valid_types = self.cognition_space_dict.get(field, {}).get('types', [])

            # Exact match
            for valid_type in valid_types:
                if valid_type.lower() == cog_type_lower:
                    return valid_type

            # Fuzzy match
            for valid_type in valid_types:
                if valid_type.lower() in cog_type_lower or cog_type_lower in valid_type.lower():
                    return valid_type

            # If no match, return first valid type
            if valid_types:
                return valid_types[0]

        # If no match found, capitalize first letter
        try:
            return cog_type_str[0].upper() + cog_type_str[1:]
        except Exception:
            return default_type

    def _normalize_cognitive_value(self, field: str, cog_type: str, cog_value: str) -> str:
        """
        Normalize cognitive value

        Args:
            field: Cognitive dimension field name
            cog_type: Normalized type
            cog_value: Original value

        Returns:
            Normalized value
        """
        # If value is empty, return type as default value
        if not cog_value:
            return cog_type

        # Convert to string and lowercase for comparison
        try:
            cog_value_str = str(cog_value).strip()
            cog_value_lower = cog_value_str.lower()
        except Exception:
            return cog_type

        # If empty after conversion, return type
        if not cog_value_lower:
            return cog_type

        # If we have a normalization map, try to use it
        if self.normalization_map and field in self.normalization_map:
            field_map = self.normalization_map.get(field, {})

            # Check for exact match in normalization map
            if cog_value_lower in field_map:
                return field_map[cog_value_lower]

            # Check for partial match
            for norm_key, norm_value in field_map.items():
                if norm_key in cog_value_lower or cog_value_lower in norm_key:
                    return norm_value

        # If we have a cognition space dictionary, try to use it
        if self.cognition_space_dict and field in self.cognition_space_dict:
            # Try to get value_list from different possible structures
            field_dict = self.cognition_space_dict.get(field, {})

            # First try to get from value_list
            value_list = field_dict.get('value_list', [])

            # If not found, try to get from values dictionary by type
            if not value_list:
                values_dict = field_dict.get('values', {})

                # If values is a dictionary, get list for this type
                if isinstance(values_dict, dict) and cog_type in values_dict:
                    value_list = values_dict.get(cog_type, [])
                # If values is a list, use it directly
                elif isinstance(values_dict, list):
                    value_list = values_dict

            # Now try to match against the value_list
            if value_list:
                # Exact match
                for valid_value in value_list:
                    if str(valid_value).lower() == cog_value_lower:
                        return valid_value

                # Fuzzy match
                for valid_value in value_list:
                    valid_value_lower = str(valid_value).lower()
                    if valid_value_lower in cog_value_lower or cog_value_lower in valid_value_lower:
                        return valid_value

                # If no match, return first valid value
                return value_list[0]

        # If no match found, capitalize first letter
        try:
            return cog_value_str[0].upper() + cog_value_str[1:]
        except Exception:
            return cog_type

    def parse_and_extract(self, content: str) -> Tuple[Dict, List[Dict], str, bool]:
        """
        Parse response and extract cognitive state, functions, and reason

        Args:
            content: LLM response content

        Returns:
            Tuple[Dict, List[Dict], str, bool]: Cognitive state, functions, reason, and success flag
        """
        # Parse response
        json_data, strategy = self.parse_response(content)

        # Extract cognitive state, functions, and reason
        cognitive_state = self.extract_cognitive_state(json_data)
        functions = self.extract_functions(json_data)
        reason = self.extract_reason(json_data)

        # Check if parsing was successful
        success = bool(json_data)

        return cognitive_state, functions, reason, success
