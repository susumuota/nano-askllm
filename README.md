# nano-askllm

Unofficial implementation of the Ask-LLM paper 'How to Train Data-Efficient LLMs', [arXiv:2402.09668](https://arxiv.org/abs/2402.09668).

[![PyPI](https://img.shields.io/pypi/v/nano-askllm?color=blue)](https://pypi.org/project/nano-askllm/)
[![GitHub License](https://img.shields.io/github/license/susumuota/nano-askllm)](https://github.com/susumuota/nano-askllm/blob/main/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/susumuota/nano-askllm)](https://github.com/susumuota/nano-askllm/commits)

<img width="514" alt="Ask-LLM prompt" src="https://github.com/susumuota/nano-askllm/assets/1632335/f7bd37dc-3702-43f9-a6db-d4f74d7822ea">

## Installation

```bash
pip install nano-askllm
```

## Usage

- Scoring C4 English dataset with `flan-t5-small` model.
> **Note**: Flan-T5 models cannot tokenize multilingual text properly (e.g. Japanese).

```python
# pip install datasets sentencepiece accelerate

from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from nano_askllm import AskLLM

model_id = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

llm = AskLLM(tokenizer, model)

batch_size = 2
num_ask = 5

for i in range(num_ask):
    datapoints = [item["text"] for item in list(dataset.take(batch_size))]
    scores = llm.ask(datapoints)
    for score, datapoint in zip(scores.tolist(), datapoints):
        text = datapoint[:40].replace("\n", " ")
        print(f"score: {score:.4f}\ttext: {text}")
    dataset = dataset.skip(batch_size)
```

- Scoring mC4 Japanese dataset with `gemma-2b-it` model. `gemma` models need to tweak the prompt template and the yes tokens.

```python
# pip install datasets sentencepiece accelerate
# hugginface-cli login

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from nano_askllm import AskLLM

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

dataset = load_dataset("allenai/c4", "ja", split="train", streaming=True)

prompt_template_prefix = "###\n"
prompt_template_postfix = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes/no
ANSWER:"""

yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]

llm = AskLLM(
    tokenizer,
    model,
    prompt_template_prefix=prompt_template_prefix,
    prompt_template_postfix=prompt_template_postfix,
    yes_tokens=yes_tokens,
    max_tokens=512,  # You can increase it up to 8192 for gemma-2b-it.
)

batch_size = 2
num_ask = 5

for i in range(num_ask):
    datapoints = [item["text"] for item in list(dataset.take(batch_size))]
    scores = llm.ask(datapoints)
    for score, datapoint in zip(scores.tolist(), datapoints):
        text = datapoint[:40].replace("\n", " ")
        print(f"score: {score:.4f}\ttext: {text}")
    dataset = dataset.skip(batch_size)
```

If you want to see the debug logs, you can set the logger as follows:

```python
from logging import DEBUG, StreamHandler, getLogger

logger = getLogger("nano_askllm.askllm")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)
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
- [x] Add examples using Hugging Face Datasets
