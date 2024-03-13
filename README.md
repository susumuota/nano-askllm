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

datapoints = [  # See tests/test_askllm.py for more examples
  "Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.",
]

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt_template = """###
{datapoint}
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes / no
ANSWER:"""

yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]

llm = AskLLM(tokenizer, model, prompt_template=prompt_template, yes_tokens=yes_tokens)
prompts = llm.get_prompts(datapoints)
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
results = llm.ask(inputs)

print(results)
```

## Development

```bash
poetry -V  # Poetry (version 1.5.1)
git clone https://github.com/susumuota/nano-askllm.git
cd nano-askllm
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

## TODO

- [ ] Add Colab notebook
- [ ] Add examples using Hugging Face Datasets
