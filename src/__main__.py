import argparse
import os
import sys

import llm_sdk
from llm_sdk import Small_LLM_Model


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

    llm: Small_LLM_Model = llm_sdk.Small_LLM_Model()

    aa = llm.encode("Hello! What is 2 plus 5?")
    print(aa)

    str_ = llm.decode([9707,    0, 3555,  374,  220,   17, 5519,  220,   20,   30])
    print(str_)
 #   bb = llm.get_logits_from_input_ids([9707,    0, 3555,  374,  220,   17, 5519,  220,   20,   30])
    vocab_path = llm.get_path_to_vocab_file()
    merges_path = llm.get_path_to_merges_file()
    tokenizer_path = llm.get_path_to_tokenizer_file()
    print("vocab_path:", vocab_path)
    print("merges_path:", merges_path)
    print("tokenizer path", tokenizer_path)


if __name__ == "__main__":
    main()
