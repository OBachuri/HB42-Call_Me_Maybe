import argparse
import os
import sys
import json
import torch

import llm_sdk
from llm_sdk import Small_LLM_Model
from src.pd_valid import CFunctions, CFunction, CPrompt


# from pydantic import BaseModel, Field, model_validator, field_validator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - "
        "Constrained Decoding LLM for Function Calling"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to write results JSON",
    )
    param = parser.parse_args()

    param._get_args()

    if not os.path.exists(param.functions_definition):
        print(f"File not found: '{param.functions_definition}'",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(param.input):
        print(f"File not found: '{param.input}'",
              file=sys.stderr)
        sys.exit(1)

    print("-"*20)
    print(param)
    print("-"*20)

    funcs_def_json = ""
    input_json = ""

    try:
        with open(param.functions_definition, "r") as defs:
            funcs_def_json = json.load(defs)
    except Exception as e:
        print(f'Error reading file "{param.functions_definition}":', e)
        sys.exit(1)

    fn = CFunctions()
    for f in funcs_def_json:
        try:
            funct = CFunction(**f)
            fn.add(funct)
        except Exception as e:
            print('Error in function definition:', f, ':', e)

    if (len(fn.fn) < 1):
        print("Error - No functions defined! Check file:",
              param.functions_definition)
        sys.exit(1)
    print("-"*20)

    try:
        with open(param.input, "r") as defs:
            input_json = json.load(defs)
    except Exception as e:
        print(f'Error reading file "{param.input}":', e)
        sys.exit(1)

    promts: list[CPrompt] = []
    for i in input_json:
        try:
            promts.append(CPrompt(**i))
        except Exception as e:
            print(f'Error in prompts "{i}":', e)

    if (len(promts) < 1):
        print("No prompts to process.")
        sys.exit(0)

    fun = ""
    f_list = ""
    max_function_name = 6
    start_for_all = "fn_"
    for f_ in fn.fn.values():
        fun = fun + f'- {f_.name}: {f_.description} \n'
        f_list = f_list + f'{f_.name}\n'
        if (len(f_.name) > max_function_name):
            max_function_name = len(f_.name)

    print("max_function_name:", max_function_name)

    llm: Small_LLM_Model = llm_sdk.Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    merges_path = llm.get_path_to_merges_file()
    tokenizer_path = llm.get_path_to_tokenizer_file()
    print("vocab_path:", vocab_path)
    print("merges_path:", merges_path)
    print("tokenizer_path:", tokenizer_path)

    llm.max_new_tokens = max_function_name
    llm.temperature = 0.0

    for promt in promts:
        str_promt = promt.prompt
        print("-"*20)
        # print(str_promt)
        str_ = f"""<|im_start|>system
You are a function selector.
Choose one function name.

Functions:
{fun}

Rules:
- Output ONLY the function name
- Do NOT output <think> </think>
- Do NOT explain
- If nothing matches, output fn_NONE
<|im_end|>

<|im_start|>user
{str_promt}
<|im_end|>

<|im_start|>assistant
"""
        print(str_)
        print("-"*10)
        aa = llm.encode(str_)
#        print(aa)

        aa_1 = aa.flatten().tolist()

        aa = llm.encode(start_for_all)
        aa_1.extend(aa.flatten().tolist())

        cc = llm.get_logits_from_input_ids(aa_1)
#    print(type(cc))
#    print(cc)

        cc_t = torch.tensor(cc)
        next_token = torch.argmax(cc_t).item()

        aa_1.append(next_token)

        for i in range(1, 20):
            cc = llm.get_logits_from_input_ids(aa_1)
            cc_t = torch.tensor(cc)
            next_token = torch.argmax(cc_t).item()
            aa_1.append(next_token)

        str_ = llm.decode(aa_1)
        print(str_)


if __name__ == "__main__":
    main()
