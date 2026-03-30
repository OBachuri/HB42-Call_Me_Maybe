*This project has been created as part of the 42 curriculum by obachuri.*

---

# Call Me Maybe  - Introduction to function calling in LLMs


This project is part of the 42.fr curriculum.

<font color="red"> *This is just the beginning of the project. Nothing works right now.* </font>
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

## License

Part of the 42 curriculum project.