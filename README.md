# nano-askllm

Unofficial implementation of the paper 'How to Train Data-Efficient LLMs' [arXiv:2402.09668](https://arxiv.org/abs/2402.09668)

[![PyPI](https://img.shields.io/pypi/v/nano-askllm?color=blue)](https://pypi.org/project/nano-askllm/)
[![GitHub License](https://img.shields.io/github/license/susumuota/nano-askllm)](https://github.com/susumuota/nano-askllm/blob/main/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/susumuota/nano-askllm)](https://github.com/susumuota/nano-askllm/commits)

## Installation

```bash
pip install nano-askllm
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nano_askllm import AskLLM

datapoints = [
  "first datapoint",  # See tests/test_askllm.py for examples
  "second datapoint",
]

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt_template = "..."  # See tests/test_askllm.py for an example
yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]

llm = AskLLM(tokenizer, model, prompt_template=prompt_template, yes_tokens=yes_tokens)
prompts = llm.get_prompts(datapoints)
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
results = llm.ask(inputs)

print(results)
```

## Development

```bash
poetry install
poetry run pytest -s     # run pytest once
poetry run -- ptw -- -s  # watch for changes and run pytest
```

## Citation

```bibtex
@misc{sachdeva2024train,
      title={How to Train Data-Efficient LLMs},
      author={Noveen Sachdeva and Benjamin Coleman and Wang-Cheng Kang and Jianmo Ni and Lichan Hong and Ed H. Chi and James Caverlee and Julian McAuley and Derek Zhiyuan Cheng},
      year={2024},
      eprint={2402.09668},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
