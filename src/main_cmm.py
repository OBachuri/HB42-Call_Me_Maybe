# import sys
# from pydantic import BaseModel, Field, model_validator, field_validator

def main() -> None:
    param = {
        "functions_definition": "data/input/functions_definition.json",
        "input": " data/input/function_calling_tests.json",
        "output": "data/output/function_calls.json"
    }
    print("-"*20)
    print(param)


if __name__ == "__main__":
    main()
