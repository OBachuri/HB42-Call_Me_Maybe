import argparse
import os
import sys
import json
from pathlib import Path
import time

import torch

import llm_sdk
from llm_sdk import Small_LLM_Model

from src.pd_valid import CFunctions, CFunction, CPrompt
from src.fn_llm_utils import LLMVocabulary
from typing import Any
import torch.nn.functional as F


# from pydantic import BaseModel, Field, model_validator, field_validator


def get_variable(llm: Small_LLM_Model,
                 v_type: str,
                 reqest_tokens_list: list, llm_vocab: LLMVocabulary) -> Any:

    str_ = ''
    value_str = ''
    next_token = 0
    value_tokens = []
    min_probability = 0.90

    # skip white spaces
    # print("skip white spaces")
    while (len(str_) == 0):
        cc = llm.get_logits_from_input_ids(reqest_tokens_list)
        # print(cc)
        cc_t = torch.tensor(cc)
        next_token = torch.argmax(cc_t).item()
        next_token_str = llm_vocab.get_str_by_token(next_token)
        # if ((next_token == 151645) or (next_token == 73594)):
        if (next_token_str is None) or (next_token_str == "```"):
            # print("t:", next_token, ", str:", llm.decode([next_token]))
            return None
        str_ = llm.decode([next_token]).strip()
        if (len(str_) == 0):
            reqest_tokens_list.append(next_token)
    # print("start variable")

    """ steps_ = list[]
        1 - allowed values or characters,
        2 - max length,
        3 - mandatory group,
        4 - 0 = skip, 1 = include in value, 2 = the end
        5 - all charsters allowed on this and next step
    """

    if (v_type == 'number') or (v_type == 'float'):

        # print("number: token=", next_token, f', s="{str_}" ,',
        #           llm_vocab.get_str_by_token(next_token))

        # str_ = llm.decode(reqest_tokens_list)
        # print(f"{str_}:---------")
        # print("+"*30)

        probs = F.softmax(cc_t, dim=-1)
        indices = list(llm_vocab.float_first)
        values = probs[indices]

        next_ = torch.argmax(values).item()
        next_token = indices[next_]
        cur_probability = values[next_].item()
        print(f"  1:{next_token:6}:'{llm_vocab.get_str_by_token(next_token)}', {cur_probability}")
        if (cur_probability > min_probability):
            value_str = llm.decode([next_token])
            reqest_tokens_list.append(next_token)
            cc = llm.get_logits_from_input_ids(reqest_tokens_list)
            cc_t = torch.tensor(cc)
        else:
            return None

        indices = list(llm_vocab.float_next)
        x = 2
        while (x < 30):
            probs = F.softmax(cc_t, dim=-1)
            values = probs[indices]
            next_ = torch.argmax(values).item()
            next_token = indices[next_]
            cur_probability = values[next_].item()
            print(f"{x:3}:{next_token:6}:'{llm_vocab.get_str_by_token(next_token)}', {cur_probability}")
            if (cur_probability < min_probability):
                break
            value_str += llm_vocab.get_str_by_token(next_token)
            reqest_tokens_list.append(next_token)
            cc = llm.get_logits_from_input_ids(reqest_tokens_list)
            cc_t = torch.tensor(cc)
            x += 1
        v = None
        try:
            if (len(value_str) > 0):
                v = float(value_str)
        except Exception:
            pass
        return (v)

    elif (v_type == 'integer'):
        # print("integer")
        probs = F.softmax(cc_t, dim=-1)
        # print(probs)
        # print("Форма:", probs.shape)        # torch.Size([3, 4, 5])
        # print("Ранг:", probs.ndim)        # 3
        # print("Размер:", probs.numel())     # 60 (3×4×5)
        # print("Тип данных:", probs.dtype)    # torch.float32
        # print("На устройстве:", probs.device) # cpu или cuda:0
        # print("Градиент:", probs.requires_grad)  # False по умолчанию

        indices = list(llm_vocab.int_first)
        values = probs[indices]
        # print(values)
        next_ = torch.argmax(values).item()
        next_token = indices[next_]
        cur_probability = values[next_].item()
        print(f"  1:{next_token:6}:'{llm_vocab.get_str_by_token(next_token)}', {cur_probability}")
        if (cur_probability > min_probability):
            value_str = llm.decode([next_token])
            reqest_tokens_list.append(next_token)
            cc = llm.get_logits_from_input_ids(reqest_tokens_list)
            cc_t = torch.tensor(cc)
        else:
            return None

        indices = list(llm_vocab.int_next)
        x = 2
        while (x < 30):
            probs = F.softmax(cc_t, dim=-1)
            values = probs[indices]
            next_ = torch.argmax(values).item()
            next_token = indices[next_]
            cur_probability = values[next_].item()
            print(f"{x:3}:{next_token:6}:'{llm_vocab.get_str_by_token(next_token)}', {cur_probability}")
            if (cur_probability < min_probability):
                break
            value_str += llm_vocab.get_str_by_token(next_token)
            reqest_tokens_list.append(next_token)
            cc = llm.get_logits_from_input_ids(reqest_tokens_list)
            cc_t = torch.tensor(cc)
            x += 1

        # probs

        # if (str_[0] in '-+0123456789'):
        #     value_str = str_[0]
        # else:
        #     probs = F.softmax(cc_t, dim=-1)
        #     value_str = str_
        # steps_ = [["-+", 1, 0, '-+0123456789'],
        #           ["0123456789", 20, 1]]
        # while (1):
        #     str_[cur_char] in steps_[cur_step][3]
        #     probs = F.softmax(cc_t, dim=-1)
        v = None
        try:
            if (len(value_str) > 0):
                v = int(value_str)
        except Exception:
            pass
        return (v)

    elif (v_type == 'string'):
        # print("read string...")
        # print("str:", str_)
        # seek to "
        i = str_.find('"')
        if (i < 0):
            aa = llm.encode('"')
            reqest_tokens_list.extend(aa.flatten().tolist())
            value_tokens.extend(aa.flatten().tolist())
        else:
            reqest_tokens_list.append(next_token)
            value_tokens.append(next_token)
            value_str = str_[(i + 1):]
        cc = llm.get_logits_from_input_ids(reqest_tokens_list)
        cc_t = torch.tensor(cc)
        next_token = torch.argmax(cc_t).item()
        str_ = llm.decode([next_token]).strip()
        x = 1
        while (x < 300):
            str_1 = llm_vocab.get_str_by_token(next_token)
            print(f"{x:3}:{next_token:6}:'{str_1}'")
            x += 1
            i = str_.find('"')
            j = 0
            if (i >= 0):
                while  (i - j - 1 >= 0) and (str_[i - j - 1] == '\\'):
                    j += 1
                if (i == j):
                    for i_s in range(0, len(value_str)):
                        if (value_str[-i_s-1] == '\\'):
                            j += 1
                        else:
                            break
                if ((j % 2) == 0):
                    reqest_tokens_list.append(next_token)
                    value_tokens.append(next_token)
                    str_ = llm.decode(value_tokens)
                    i = str_.find('"')
                    j = str_.rfind('"')
                    # print("---end of variable---")
                    str_ = str_[i+1:j]
                    try:
                        str_ = str_.encode('utf-8').decode('unicode_escape')
                    except Exception:
                        pass
                    return (str_)
            value_str += str_
            reqest_tokens_list.append(next_token)
            value_tokens.append(next_token)
            cc = llm.get_logits_from_input_ids(reqest_tokens_list)
            cc_t = torch.tensor(cc)
            next_token = torch.argmax(cc_t).item()
            next_token_str = llm_vocab.get_str_by_token(next_token)
            # if ((next_token == 151645) or (next_token == 73594)):
            if (next_token_str is None) or (next_token_str == "```"):
                print("t:", next_token, ", str:", next_token_str)
                str_ = llm.decode(value_tokens)
                i = str_.find('"')
                str_ = str_[i+1:].encode('utf-8').decode('unicode_escape')
                return (str_)
            str_ = llm.decode([next_token]).strip()
        str_ = llm.decode(value_tokens)
        i = str_.find('"')
        str_ = str_[i+1:].encode('utf-8').decode('unicode_escape')
        return (str_)
    elif (v_type == 'boolean'):
        steps_ = [
                  [['true', 'false'], 0, 0, 1],
                  ]

    return None


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
    output_json = []

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
    # start_for_all = "fn"
    for f_ in fn.fn.values():
        fun = fun + f'- {f_.name}: {f_.description} \n'
        f_list = f_list + f'{f_.name}\n'
        if (len(f_.name) > max_function_name):
            max_function_name = len(f_.name)

    print("max_length_of_function_name:", max_function_name)

    llm: Small_LLM_Model = llm_sdk.Small_LLM_Model()

    # llm: Small_LLM_Model = llm_sdk.Small_LLM_Model(
    #     model_name="Qwen/Qwen3-1.7B")

    vocab_path = llm.get_path_to_vocab_file()
    merges_path = llm.get_path_to_merges_file()
    tokenizer_path = llm.get_path_to_tokenizer_file()
    print("vocab_path:", vocab_path)
    print("merges_path:", merges_path)
    print("tokenizer_path:", tokenizer_path)

    try:
        llm_vocab = LLMVocabulary(vocab_path)
    except Exception as e:
        print("Error reading model Vocabulary file:", e)
        sys.exit(1)

    # str_1 = llm_vocab.get_str_by_token(151645)
    # print("stop:", str_1, type(str_1))
    # str_1 = llm_vocab.get_str_by_token(73594)
    # print(f'str_end_of_json:"{str_1}"', type(str_1))

    # print(dir(llm))
    print("="*10)

    # llm.max_new_tokens = max_function_name
    # llm.temperature = 0.1
    # llm.verbose=False

    # 1. Record start time
    start_time = time.perf_counter()

    prompt_number = 0
    for promt in promts:
        str_promt = promt.prompt
        prompt_number += 1
        print("="*30, f"[{prompt_number}/{len(promts)}]")
        print("Prompt:", str_promt)
#         str_ = f"""<|im_start|>system
# You are a direct assistant.  Non-thinking mode.
# You are a function selector.
# Choose one function name.

# Functions:
# {fun}

# Rules:
# - Output ONLY the function name
# - If nothing matches, output: fn_NONE
# - Answer directly
# - No thinking steps
# - No explanations
# <|im_end|>

# <|im_start|>user
# {str_promt}
# <|im_end|>

# <|im_start|>assistant
# """

        str_ = f"""<|im_start|>system
You are a function selector.
Choose one function name.

Functions:
{fun}

Rules:
- Output ONLY the function name
- If nothing matches, output: fn_NONE
<|im_end|>

<|im_start|>user
{str_promt}
<|im_end|>

<|im_start|>assistant
"""

        # str_ = (
        #     "<|im_start|>system\nYou are a reliable function-calling "
        #     "assistant. "
        #     f"You have access to the following functions:\n"
        #     f"{fun}\n\n"
        #     "You must output ONLY a valid JSON object representing a "
        #     "function call. Start exactly with {\"fn_name\":...\n"
        #     "CRITICAL: If you need to write a regular expression,"
        #     " DO NOT use JavaScript "
        #     "regex delimiters like /.../g."
        #     " Just output the raw Python pattern. "
        #     "If your pattern requires backslashes"
        #     " (like \\w or \\b), you MUST double-escape "
        #     "them (e.g. \\\\w) because this is a JSON string.<|im_end|>\n"
        #     f"<|im_start|>user\n{str_promt}<|im_end|>\n"
        #     "<|im_start|>assistant\n"
        # )

        # print(str_)
        aa = llm.encode(str_)
#        print(aa)

        aa_1 = aa.flatten().tolist()

        # aa = llm.encode(start_for_all)
        # aa_1.extend(aa.flatten().tolist())

        # str_ = llm.decode([2236])
        # print(str_, "------")

        mylist = []
        aa = llm.encode('</think>')
        aa_1.extend(aa.flatten().tolist())
        # aa = llm.encode(start_for_all)
        # print(aa)
        # print("--fn--")
        # aa_1.extend(aa.flatten().tolist())
        # mylist.extend(aa.flatten().tolist())


        cc = llm.get_logits_from_input_ids(aa_1)
#    print(type(cc))
#    print(cc)

        cc_t = torch.tensor(cc)
        next_token = torch.argmax(cc_t).item()
        aa_1.append(next_token)

        for i_ in range(1, max_function_name + 1):
            # print("---ccccc:")
            # print(aa_1)
            cc = llm.get_logits_from_input_ids(aa_1)
            cc_t = torch.tensor(cc)
            next_token = torch.argmax(cc_t).item()
            next_token_str = llm_vocab.get_str_by_token(next_token)
            if next_token_str is None:
                break
            # if (next_token == 151645):
            #     break
            # print("t:", next_token, ", str:", llm.decode([next_token]),"s:",next_token_str,type(next_token_str))
            aa_1.append(next_token)
            mylist.append(next_token)
            i_ += 1

        str_ = llm.decode(mylist)
        fn_name = str_.strip()
        print("Function:", fn_name)
        fn_c = fn.fn.get(fn_name, None)
        if (fn_c is None):
            continue
        s_ = ', '.join(f"'{p}': '{fn_c.parameters[p].type}'" for p in fn_c.parameters.keys())
        fn_param_list = [[str(p), fn_c.parameters[p].type] for p in fn_c.parameters.keys()]
        if (len(fn_param_list) < 1):
            continue
        print("Parameters list:", fn_param_list)
        # prompt_number += 1
        fn_c.parameters.keys()
        str_ = f"""<|im_start|>system
You are a direct assistant. You are a strict parameter extractor.
Write JSON with function parameters.

Function: {fn_name}
Description: {fn_c.description}
Parameters {{name: type}}: {{{s_}}}

Rules:
- Return ONLY JSON with parameters for the function
- Do NOT return result
- You are NOT executing the function. You are ONLY preparing the input parameters.
<|im_end|>

<|im_start|>user
{str_promt}
<|im_end|>

<|im_start|>assistant
"""

        # print(str_)
        # print("*"*10)
        # print(mylist)

        aa = llm.encode(str_)
        aa_1 = aa.flatten().tolist()
        mylist = []
        f_param: dict[str, Any] = {}
        aa = llm.encode('</think>')
        aa_1.extend(aa.flatten().tolist())

#        cc = llm.get_logits_from_input_ids(aa_1)
        aa = llm.encode('```json')
#        print(aa)
        aa_1.extend(aa.flatten().tolist())
        aa_1.append(llm_vocab.get_token_by_str('Ċ'))

        # # Print model answer with out constraction decoding
        # aa_2 = []
        # aa_2.extend(aa_1)
        # for i_ in range(1, 50):
        #     cc = llm.get_logits_from_input_ids(aa_2)
        #     cc_t = torch.tensor(cc)
        #     next_token = torch.argmax(cc_t).item()
        #     if ((next_token == 151645) or (next_token == 73594)):
        #         # print("t:", next_token, ", str:", llm.decode([next_token]))
        #         break
        #     print("t:", next_token,
        #           f', str:"{llm_vocab.get_str_by_token(next_token)}"')
        #     aa_2.append(next_token)
        # print("#"*30)
        # print(llm.decode(aa_2))
        # print("#"*30)

        for i_param in range(0, len(fn_param_list)):
            type_param = fn_param_list[i_param][1]
            name_param = fn_param_list[i_param][0]
            if (i_param == 0):
                str_param = f'{{"{name_param}":'
            else:
                str_param = f', "{name_param}":'
            if (type_param == "string") and (fn_name.find("regex") < 0):
                str_param += " "
            # cc = llm.get_logits_from_input_ids(aa_1)
            # aa = llm.encode(f'{{"{fn_param_list[0][0]}": ')
            # print(aa.flatten().tolist())
            aa = llm.encode(str_param)
            aa_1.extend(aa.flatten().tolist())
            mylist.extend(aa.flatten().tolist())

            # print("*"*10)
            # str_ = llm.decode(aa_1)
            # print(str_)
            # print("*"*10)
            print(f"get_variable (name={fn_param_list[i_param][0]}, type={type_param})")
            v = get_variable(llm, type_param, aa_1, llm_vocab)
            f_param[name_param] = v
            print(f'-- variable value: "{v}", type:', type(v))

            # str_ = llm.decode(aa_1)
            # print(str_)
            # print("*"*10)

            # cc_t = torch.tensor(cc)
            # next_token = torch.argmax(cc_t).item()

            # aa_1.append(next_token)

            # for i_ in range(1, 50):
            #     cc = llm.get_logits_from_input_ids(aa_1)
            #     cc_t = torch.tensor(cc)
            #     next_token = torch.argmax(cc_t).item()
            #     if ((next_token == 151645) or (next_token == 73594)):
            #         # print("t:", next_token, ", str:", llm.decode([next_token]))
            #         break
            #     # print("t:", next_token, ", str:", llm.decode([next_token]))
            #     aa_1.append(next_token)
            #     mylist.append(next_token)

        # print(next_token, ", l:", aa_1[-3:])
        # str_ = llm.decode(mylist)
        # idx = str_.rfind("}")
        # if idx != -1:
        #     str_ = str_[:idx + 1]
        # print("Parameters values:", str_)
        try:
            # fn_param_json = json.loads(str_)
            output_json.append({
                "prompt": str_promt,
                "name": fn_name,
                "parameters": f_param})
        except Exception as e:
            print("Error converting parameters to json:", e)
        print(output_json[-1])

    # Record end time
    end_time = time.perf_counter()

    # Calculate duration
    duration = int(end_time - start_time)
    print(f"Execution time: {duration // 60}:{duration % 60} seconds")

    # print(output_json)

    out_path = Path(param.output)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as out_file:
            json.dump(output_json, out_file, indent=2)
    except Exception as e:
        print(f"Error by writting json in file {param.output}:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
