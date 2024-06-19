# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from logging import ERROR, StreamHandler, getLogger
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from nano_askllm import AskLLM
import wandb


# Set logging level to ERROR.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(ERROR)
handler = StreamHandler()
handler.setLevel(ERROR)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Scoring by Ask-LLM.")
parser.add_argument("--start", type=int, help="Start index of the dataset.", default=0)
parser.add_argument("--end", type=int, help="End index of the dataset.", default=-1)
parser.add_argument("--output_tsv", type=str, help="Output TSV filename.", default="output")
parser.add_argument("--batch_size", type=int, help="Batch size.", default=8)
parser.add_argument("--max_tokens", type=int, help="Max tokens for Ask-LLM.", default=512)
parser.add_argument("--dataset_path", type=str, help="Dataset path name.", default="allenai/c4")
parser.add_argument("--dataset_lang", type=str, help="Dataset language.", default="ja")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--dataset_fields", type=str, help="Dataset fields separated by comma.", default="text")
parser.add_argument("--model_id", type=str, help="Model ID.", default="Rakuten/RakutenAI-7B-instruct")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--log_interval", type=int, help="Log interval.", default=1000)
parser.add_argument("--wandb_project", type=str, help="WandB project name.", default=None)
parser.add_argument("--wandb_entity", type=str, help="WandB entity name.", default=None)
parser.add_argument("--wandb_name", type=str, help="WandB experiment name name.", default=None)
args = parser.parse_args()

start = args.start
end = args.end
batch_size = args.batch_size
max_tokens = args.max_tokens
output_tsv = args.output_tsv
dataset_path = args.dataset_path
dataset_lang = args.dataset_lang
dataset_split = args.dataset_split
dataset_fields = args.dataset_fields.split(",")
model_id = args.model_id
cache_dir = args.cache_dir
log_interval = args.log_interval
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_name = args.wandb_name

if wandb_project is not None and wandb_entity is not None and wandb_name is not None:
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name)

tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", cache_dir=cache_dir, attn_implementation="flash_attention_2"
)

dataset = load_dataset(dataset_path, name=dataset_lang, split=dataset_split, cache_dir=cache_dir)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

prompt_template_prefix = "###\n"
prompt_template_postfix = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes / no
ANSWER:"""  # noqa: E501

# yes_tokens = ["yes", "Yes"]  # for RakutenAI-7B-instruct
yes_tokens = ["yes", "Yes", "YES"]  # for Phi-3

llm = AskLLM(
    tokenizer,
    model,
    prompt_template_prefix=prompt_template_prefix,
    prompt_template_postfix=prompt_template_postfix,
    yes_tokens=yes_tokens,
    max_tokens=max_tokens,
)

end = len(dataset) if end == -1 else end
end = min(end, len(dataset))

# Resume from the last line of the output file.
try:
    with open(output_tsv, "r") as file:
        last_line = file.readlines()[-1]
        last_index = last_line.split("\t")[1]
        start = int(last_index) + 1 if len(last_index) > 0 else start
except FileNotFoundError:
    pass
except IndexError:
    pass

print(f"Scoring from {start} to {end} with batch size {batch_size}.", flush=True)

with open(output_tsv, "a") as f:
    batch_count = 0
    for i in range(start, end, batch_size):
        batch_start = i
        batch_end = i + batch_size if i + batch_size < end else end
        data = dataset[batch_start:batch_end]
        if len(data) == 0:
            break
        dicts = pd.DataFrame(data).to_dict(orient="records")
        batch = ["\n".join([dic[field] or "" for field in dataset_fields]) for dic in dicts]
        scores = llm.ask(batch)
        for index, score in enumerate(scores.tolist()):
            line = f"{score:.4f}\t{batch_start + index}"
            f.write(line + "\n")
            f.flush()
            # print(line, flush=True)
        if wandb_project is not None and wandb_entity is not None and batch_count % log_interval == 0:
            wandb.log({"score": scores.tolist()[0]}, step=batch_start)
        del scores
        batch_count += 1

if wandb_project is not None and wandb_entity is not None and wandb_name is not None:
    wandb.finish()
