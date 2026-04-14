*This project has been created as part of the 42 curriculum by obachuri.*

---

# Call Me Maybe  - Introduction to function calling in LLMs


This project is part of the 42.fr curriculum.

---

## Description

Function calling system using Qwen3-0.6B LLM with constrained decoding. Translates natural language prompts into structured function calls with typed parameters. Achieves 100% valid JSON output through hybrid approach: LLM-driven function selection combined with type-constrained value extraction.

## Instructions

### Installation

**Python 3.10+** and **uv** must be installed in avance.

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
- [HuggingFace description for Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Pydantic Docs](https://docs.pydantic.dev/latest/)


## Algorithm and implementation

### In general, the LLM generation process includes the following steps:

#### 1) **Tokenization** 
The string with Prompt trasforms in array of token Id`s. A Token is the smallest unit of text (a word, part of a word, character or punctuation) that an AI model processes as a single numerical value.
```
myTokenlist = llm.encode(str_prompt).flatten().tolist()
```

str_prompt = "Hello word!"
myTokenlist = [9707, 3409, 0]

| token_id | value |
| :--- | :--- |
| 9707 | "Hello" |
| 3409 | "Ġword" |
| 0 | "!" |

Here 'Ġ' - spase, 'Ċ' - new line.

We have access to the file with LLM vocabulary:
```
file_with_llm_vocab_path_and_name = llm.get_path_to_vocab_file()
with open(file_with_llm_vocab_path_and_name, "r", encoding="utf-8") as f:
    llm_vocab: dict[str, int] = json.load(f)
```

#### 2) **LLM Processing**

As result of LLM Processing we gets Logits. 
```
logits = llm.get_logits_from_input_ids(myTokenlist)
```
Logits are the raw, unnormalized output values produced by the last linear layer of a neural network of LLM.
It can be any real number, from \(-\infty \) to \(+\infty \).
Logits show probability scores for each possible next token.
A higher logit value relative to others means that the value is more likely to be in the answer.
The position of logit in the Logits array corresponds to the Token ID. 
But quantity of logits could be more then quantity of tokens in LLM vocabulary. 
If the logit with the maximum value is beyond LLM vocabulary, it shows the end of LLM answer.

To go from logits to probability of tokens we need to scales the raw values so that they all fall between 0 and 1 and sum up to 1.
In PyTorch, this is typically done using torch.softmax():
```
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
```

#### 3) **Token Selection**

If Greedy Decoding used, the next token is the token with the maximum logit value.
```
    next_token = int(torch.argmax(torch.tensor(logits)).item())
```
When used the Constrained decoding method, the token with the highest probability is selected among the allowed tokens at the current step.

Add the selected token to myTokenlist.
```
    myTokenlist.append(next_token)
```
After that and go to step 2) 

Repeat steps 2 and 3 to get the complete answer.


###  **1 - Function selection**

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
Here is used Greedy Decoding.

###  **2 - Parameters of the function**

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

Next, for every parameters : 
1) Add name of parameters to list of Token IDs, like this
```Python
myTikenlist.extend(llm.encode(f'"{paramer_name}"').flatten().tolist())
```
2) Run Constrained decoding for data type of parameter

### **3 - Constrained decoding values**

Constrained decoding is used to extract parameter values with strict type control.
Instead of allowing the model to generate arbitrary tokens, we restrict the set of valid tokens at each decoding step.

In this project Constrained decoding implemented for following types of data: number, integer, string, boolean.

**Integer decoding**

The first character must be one of this: "+-0123456789" (```llm_vocab.int_first: set[int]```), nexts could be only degits (```llm_vocab.int_next: set[int]```).

A token is accepted only if its probability exceeds a threshold:
```
value_str: str = ''
indices = list(llm_vocab.int_first)

values = probs[indices]
next_ = int(torch.argmax(values).item())
next_token = indices[next_]
cur_token_probability = values[next_].item()

if cur_token_probability > min_probability:
  value_str += llm.decode([next_token])
else:
  # the end 

....

value = int(value_str)
```

**Number (float) decoding**

The first character must be one of this: "+-.0123456789" (```llm_vocab.float_first: set[int]```)
Nexts could be: ".eE0123456789" (```llm_vocab.float_next: set[int]```)

**String decoding**

Strings are handled differently:
- Generation starts after detecting or forcing "
- Tokens are appended until an unescaped closing quote is found
- Escape sequences (```\", \\```) are handled manually

This ensures valid JSON-compatible strings.

**Boolean decoding**

Boolean values are inferred from the first generated token:
- "T" or non-zero digit → True
- "F" or "0" → False
Then the full token sequence (True / False) is appended.

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

Manual testing on 11 provided test - all passing.

**Edge cases covered**
- large numbers (300000, 3e5)
- negative values
- floating-point numbers
- quoted strings with escapes
- boolean values



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

**Run** 

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