"""
ASCE社交代理Camel响应模式定义

该模块定义了用于Camel框架的Pydantic模型，用于结构化LLM响应。
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, RootModel


class CognitiveField(BaseModel):
    """认知维度字段模型"""
    type: str = Field(..., description="Type of cognitive dimension")
    value: str = Field(..., description="Value of cognitive dimension")


class Opinion(BaseModel):
    """观点模型"""
    viewpoint_1: str = Field(..., description="Support level for viewpoint 1")
    viewpoint_2: str = Field(..., description="Support level for viewpoint 2")
    viewpoint_3: str = Field(..., description="Support level for viewpoint 3")
    viewpoint_4: str = Field(..., description="Support level for viewpoint 4")
    viewpoint_5: str = Field(..., description="Support level for viewpoint 5")
    viewpoint_6: str = Field(..., description="Support level for viewpoint 6")


class CognitiveState(BaseModel):
    """认知状态模型"""
    mood: CognitiveField = Field(..., description="mood dimension")
    emotion: CognitiveField = Field(..., description="Emotion dimension")
    stance: CognitiveField = Field(..., description="Stance dimension")
    thinking: CognitiveField = Field(..., description="Thinking dimension")
    intention: CognitiveField = Field(..., description="Intention dimension")


class FunctionArguments(RootModel):
    """函数参数模型，使用RootModel允许任意参数"""
    root: Dict[str, Any] = Field(..., description="Function arguments")


class Function(BaseModel):
    """函数调用模型"""
    name: str = Field(..., description="Function name")
    arguments: Dict[str, Any] = Field(..., description="Function arguments")


class ASCEResponse(BaseModel):
    """ASCE响应模型，定义完整的响应结构"""
    reason: str = Field(..., description="Reasoning and explanation for actions")
    cognitive_state: CognitiveState = Field(..., description="Cognitive state dimensions")
    opinion: Opinion = Field(..., description="Opinion on viewpoints")
    functions: List[Function] = Field(..., description="Function calls list")
