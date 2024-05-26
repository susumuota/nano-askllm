# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from logging import ERROR, StreamHandler, getLogger
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from nano_askllm import AskLLM
import wandb


# for https://huggingface.co/datasets/llm-jp/hh-rlhf-12k-ja dataset
# https://github.com/llm-jp/llm-jp-dpo/blob/020fa2eada0951929380811835ae6cc6b1cd84b3/train.py#L16
def return_prompt_and_responses(samples) -> dict[str, list[str]]:
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    for conversation, chosen, rejected in zip(samples["conversations"], samples["chosen"], samples["rejected"]):
        prompt: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
        for utterance in conversation:
            if utterance["from"] == "human":
                prompt += f"\n\n### 指示:\n{utterance['value']}"
            else:
                prompt += f"\n\n### 応答:\n{utterance['value']}"
        prompt += "\n\n### 応答:\n"
        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


# Set logging level to ERROR.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(ERROR)
handler = StreamHandler()
handler.setLevel(ERROR)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Scoring by Ask-LLM.")
parser.add_argument("--rank", type=int, help="Rank of the process.", default=0)
parser.add_argument("--start", type=int, help="Start index of the dataset.", default=0)
parser.add_argument("--end", type=int, help="End index of the dataset.", default=-1)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=8)
parser.add_argument("--max_tokens", type=int, help="Max tokens for Ask-LLM.", default=512)
parser.add_argument("--output_basename", type=str, help="Output file basename.", default="output")
parser.add_argument("--output_suffix_offset", type=int, help="Output file suffix offset.", default=0)
parser.add_argument("--dataset_path", type=str, help="Dataset path name.", default="allenai/c4")
parser.add_argument("--dataset_lang", type=str, help="Dataset language.", default="ja")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--dataset_fields", type=str, help="Dataset fields separated by comma.", default="text")
parser.add_argument("--model_id", type=str, help="Model ID.", default="Rakuten/RakutenAI-7B-instruct")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--log_interval", type=int, help="Log interval.", default=1000)
parser.add_argument("--wandb_project", type=str, help="WandB project name.", default=None)
parser.add_argument("--wandb_entity", type=str, help="WandB entity name.", default=None)
args = parser.parse_args()

rank = args.rank
start = args.start
end = args.end
batch_size = args.batch_size
max_tokens = args.max_tokens
output_basename = args.output_basename
output_suffix_offset = args.output_suffix_offset
dataset_path = args.dataset_path
dataset_lang = args.dataset_lang
dataset_split = args.dataset_split
dataset_fields = args.dataset_fields.split(",")
model_id = args.model_id
cache_dir = args.cache_dir
log_interval = args.log_interval
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity

if wandb_project is not None and wandb_entity is not None:
    wandb.init(project=wandb_project, entity=wandb_entity)

tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto", device_map=f"cuda:{rank}", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map=f"cuda:{rank}", cache_dir=cache_dir
)
dataset = load_dataset(dataset_path, name=dataset_lang, split=dataset_split, cache_dir=cache_dir)

model.generation_config.pad_token_id = model.generation_config.eos_token_id

if all(column in dataset.column_names for column in ["conversations", "chosen", "rejected"]):
    dataset = dataset.map(return_prompt_and_responses, batched=True, remove_columns=dataset.column_names)
    dataset_fields = ["prompt" if field == "conversations" else field for field in dataset_fields]
    print(f"Dataset columns: {dataset.column_names}", flush=True)

prompt_template_prefix = "###\n"
prompt_template_postfix = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for post-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world.

OPTIONS: yes / no
ANSWER:"""  # noqa: E501

yes_tokens = ["yes", "Yes"]  # for RakutenAI-7B-instruct

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
suffix = output_suffix_offset + rank
filename = f"{output_basename}_{suffix}.tsv"
try:
    with open(filename, "r") as file:
        last_line = file.readlines()[-1]
        last_index = last_line.split("\t")[1]
        start = int(last_index) + 1 if len(last_index) > 0 else start
except FileNotFoundError:
    pass
except IndexError:
    pass

print(f"Scoring from {start} to {end} with batch size {batch_size}.", flush=True)

with open(filename, "a") as f:
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
            wandb.log({"score": scores.tolist()[0], "index": batch_start})
        del scores
        batch_count += 1

if wandb_project is not None and wandb_entity is not None:
    wandb.finish()
