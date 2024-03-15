# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from logging import DEBUG, StreamHandler, getLogger
from time import time

import torch
from datasets import load_dataset
from transformers import (  # noqa: F401
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from nano_askllm import AskLLM, __version__

# Set logging level to DEBUG.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


def test_version():
    assert __version__ == "0.2.0"
    print("test_version passed")


def test_flan_t5_c4_en():
    # load the model and tokenizer
    # *Flan T5 only works on English datasets.*
    model_id = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)

    # Load C4 English dataset.
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/en
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    llm = AskLLM(tokenizer, model)
    assert isinstance(llm, AskLLM)

    batch_size = 1
    num_ask = 10

    print("-" * 80)
    start_time = time()
    for i in range(num_ask):
        print(f"batch {i + 1} start")
        datapoints = [item["text"] for item in list(dataset.take(batch_size))]
        results = llm.ask(datapoints)
        assert isinstance(results, torch.Tensor) and results.shape == (batch_size,)
        assert all(results >= 0.0) and all(results <= 1.0)
        for score, datapoint in zip(results.tolist(), datapoints):
            text = datapoint[:80].replace("\n", " ")
            print(f"score: {score:.4f}\ttext: {text}")
        del results
        dataset = dataset.skip(batch_size)
        end_time = time()
        print(f"batch {i + 1} end, {(end_time - start_time):.4f} seconds")
        print("-" * 80)
        start_time = end_time

    del llm, dataset, model, tokenizer
    print("test_flan_t5_c4_en passed")


def test_gemma_mc4_ja():
    # load the model and tokenizer
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # For 4bit quantization on Colab T4 GPU
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)

    # Load mC4 Japanese dataset.
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/ja
    dataset = load_dataset("allenai/c4", "ja", split="train", streaming=True)

    # Default prompt template is not suitable for gemma-2b-it.
    # I changed "OPTIONS:" format from "\n- yes\n- no\n" to " yes/no\n".
    # I added "ANSWER:" to the last line to increase the probability of "yes" or "no" being the first token.
    # TODO: prompt engineering is necessary for each model.
    prompt_template_prefix = "###\n"
    prompt_template_postfix = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes/no
ANSWER:"""  # noqa: E501

    yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]  # for gemma-2b-it

    llm = AskLLM(
        tokenizer,
        model,
        prompt_template_prefix=prompt_template_prefix,
        prompt_template_postfix=prompt_template_postfix,
        yes_tokens=yes_tokens,
        max_tokens=512,  # You can increase it up to 8192 for gemma-2b-it.
    )
    assert llm is not None

    batch_size = 1
    num_ask = 10

    print("-" * 80)
    start_time = time()
    for i in range(num_ask):
        print(f"batch {i + 1} start")
        datapoints = [item["text"] for item in list(dataset.take(batch_size))]
        results = llm.ask(datapoints)
        assert isinstance(results, torch.Tensor) and results.shape == (batch_size,)
        assert all(results >= 0.0) and all(results <= 1.0)
        for score, datapoint in zip(results.tolist(), datapoints):
            text = datapoint[:80].replace("\n", " ")
            print(f"score: {score:.4f}\ttext: {text}")
        del results
        dataset = dataset.skip(batch_size)
        end_time = time()
        print(f"batch {i + 1} end, {(end_time - start_time):.4f} seconds")
        print("-" * 80)
        start_time = end_time

    del llm, dataset, model, tokenizer
    print("test_gemma_mc4_ja passed")
