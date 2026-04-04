"""Pydantic models for function calling data structures."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, field_validator


class ParameterDefinition(BaseModel):
    """Defines a single parameter for a function."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that type is a known JSON Schema type."""
        allowed = {"number", "string", "boolean", "integer", "array", "object", "null"}
        if v not in allowed:
            raise ValueError(f"Unknown type '{v}'. Allowed: {allowed}")
        return v


class ReturnDefinition(BaseModel):
    """Defines the return type of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Describes a callable function with its parameters and return type."""

    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition


class Prompt(BaseModel):
    """A single natural language prompt to process."""

    prompt: str


class FunctionCall(BaseModel):
    """Result of resolving a prompt to a function call."""

    prompt: str
    name: str
    parameters: Dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def validate_parameters_not_none(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters dict is not None."""
        if v is None:
            return {}
        return v


class AppConfig(BaseModel):
    """Application configuration from CLI arguments."""

    functions_definition: str
    input: str
    output: str
    model_name: Optional[str] = "Qwen/Qwen3-0.6B"
