*This project has been created as part of the 42 curriculum by obachuri.*

---

# Call Me Maybe  - Introduction to function calling in LLMs


This project is part of the 42.fr curriculum.

---

## Description

Function calling system using Qwen3-0.6B LLM with constrained decoding. Translates natural language prompts into structured function calls with typed parameters. Achieves 100% valid JSON output through hybrid approach: LLM-driven function selection combined with type-constrained value extraction.

## Instructions

### Installation

```bash
make install
```

### Execution

```bash
make run
```

Run with custom paths:
```bash
uv run python -m src --functions_definition <functions.json> --input <tests.json> --output <results.json>
```

Config files example provided in [Example Usage](#example-usage) and in folder *data*.

## Resources

- [Prompt stucture and Control Tokens for Qwen3 LLM](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Pydantic Docs](https://docs.pydantic.dev/latest/)


## Algorithm and implementation

**1 - Function name**

The first step is to get the function name.

The Prompt for select the function formed using control tokens looks like this:
```
<|im_start|>system
You are a function selector.
Choose one function name.

Functions:
{list_of_function_name_and_description}

Rules:
- Output ONLY the function name
- If nothing matches, output: fn_NONE
<|im_end|>

<|im_start|>user
{str_user_prompt}
<|im_end|>

<|im_start|>assistant
</think>
```
The first word in model answer must be a function name, but it must be checked against a list of given function names. 

**2 - Parameters of the function**

The Prompt to get parameters values:
```
<|im_start|>system
You are a direct assistant. You are a strict parameter extractor.
Write JSON with function parameters.

Function: {function_name}
Description: {function_description}
Parameters {{name: type}}: {function_parameters_json}

Rules:
- Return ONLY JSON with parameters for the function
- Do NOT return result
- You are NOT executing the function. You are ONLY preparing the input parameters.
<|im_end|>

<|im_start|>user
{str_user_prompt}
<|im_end|>

<|im_start|>assistant
</think>```json
{
```
**3 - Constrained decoding values**


## Performance

- Function selection: 100%
- Number extraction: 99%  - *It work for given examples, but for float with "e" sometimes give wrong values (example: .1e3 could be evaluated as 1)*  
- String extraction: 100%
- Regex generation: 99% - *Sometimes put extra spaces or back slashes (\\)*
- JSON validity: 100% 

Test platform: CPU - Intel i7-13700, RAM - 16 Gb DDR4  (model run on CPU)
Performance: ~8 sec/prompt

## Challenges

## Testing

## Example Usage

**Functions definition** (`data/input/functions_definition.json`):
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }
]
```

**Input** (`data/input/function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3e5?"}
]
```

**run** 

```bash
make run
```

**Output** (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3e5?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 2.0,
      "b": 300000.0
    }
  }
]
```


## License

Part of the 42 curriculum project.