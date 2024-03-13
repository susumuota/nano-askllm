# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from logging import DEBUG, StreamHandler, getLogger
from time import time

from datasets import load_dataset
# import torch
from transformers import (  # noqa: F401
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import nano_askllm

# Set logging level to DEBUG.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


def test_version():
    assert nano_askllm.__version__ == "0.1.0"
    print("test_version passed")


def test_askllm():
    # load the model and tokenizer
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # for 4bit quantization on Colab T4 GPU
    # model_id = "google/gemma-2b-it"
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)

    # for Flan T5 model
    # model_id = "google/flan-t5-large"
    # tokenizer = T5Tokenizer.from_pretrained(model_id)
    # model = T5ForConditionalGeneration.from_pretrained(model_id)

    # Load C4 English (or mC4 Japanese) dataset.
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # dataset = load_dataset("allenai/c4", "ja", split="train", streaming=True)
    #
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/en
    # https://huggingface.co/datasets/allenai/c4/viewer/ja

    # Default prompt template is not suitable for gemma-2b-it.
    # I changed "OPTIONS:" format from "\n- yes\n- no\n" to " yes / no".
    # I added "ANSWER:" to the last line to increase the probability of "yes" or "no" being the first token.
    # TODO: prompt engineering is necessary for each model.
    prompt_template = """###
{datapoint}
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes / no
ANSWER:"""  # noqa: E501

    # Typical yes tokens, depending on the model.
    yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]  # for gemma-2b-it
    # yes_tokens = ["yes", "Yes"]  # for flan-t5-large

    llm = nano_askllm.AskLLM(tokenizer, model, prompt_template=prompt_template, yes_tokens=yes_tokens)
    assert llm is not None

    batch_size = 1
    num_ask = 5

    start_time = time()
    for i in range(num_ask):
        print(f"batch {i + 1} start")
        datapoints = [item["text"] for item in list(dataset.take(batch_size))]
        prompts = llm.get_prompts(datapoints)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        results = llm.ask(inputs)
        for score, datapoint in zip(results.tolist(), datapoints):
            text = datapoint[:80].replace("\n", " ")
            print(f"score: {score:.4f}\ttext: {text}")
        del inputs, results
        dataset = dataset.skip(batch_size)
        end_time = time()
        print(f"batch {i + 1} end, {(end_time - start_time):.4f} seconds")
        print("-" * 80)
        start_time = end_time

    print("test_askllm passed")
