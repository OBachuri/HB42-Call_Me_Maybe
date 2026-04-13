import re
from pydantic import BaseModel, Field, field_validator
# , ValidationError , model_validator


class CParameterType(BaseModel):
    """
    Single parameter or return value type in a function definition.
    """
    type: str

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        allowed_types = {"number", "string", "boolean", "integer",
                         "float", "array", "object"}
        if v not in allowed_types:
            raise ValueError(
                f"Unsupported type '{v}'."
                " Type must be one of:", allowed_types)
        return v


class CFunction(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=5)
    parameters: dict[str, CParameterType] = {}
    returns: CParameterType

    @field_validator("name")
    def name_check(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Name of the function must not be empty!")
        if not bool(re.match(r'^[A-Za-z][A-Za-z0-9_]*$', v)):
            raise ValueError('Function name must begin from a letter and '
                             'contain only letters, numbers, or "_".'
                             f"(name={v})")
        return v

    @field_validator("description")
    def name_description(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Description of the function must not be empty!")
        return v

    # @field_validator("returns")
    # def returns_check(cls, v):
    #     if (len(v) < 1):
    #         raise ValueError("Returns value type of the"
    #                          " function must be defined!")
    #     return v


class CFunctions(BaseModel):
    fn: dict[str, CFunction] = {}

    def add(self, f: CFunction) -> None:
        if not (self.fn.get(f.name, None) is None):
            raise ValueError("Name of the function must be unique!"
                             f"(name='{f.name}')")
        self.fn[f.name] = f


class CPrompt(BaseModel):
    prompt: str = Field(min_length=5)

    @field_validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        v = v.strip()
        if (len(v) < 5):
            raise ValueError("Prompt cannot be empty.")
        return v
