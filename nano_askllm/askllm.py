# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

logger = logging.getLogger(__name__)


class AskLLM:
    # A prompt template. See https://arxiv.org/abs/2402.09668 page 4.
    DEFAULT_PROMPT_TEMPLATE = """###
{datapoint}
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS:
- yes
- no
"""  # noqa: E501

    # Each of the word must be tokenized to a single token.
    DEFAULT_YES_TOKENS = ["yes", "Yes"]

    def __init__(
        self,
        tokenizer: AutoTokenizer | T5Tokenizer,
        model: AutoModelForCausalLM | T5ForConditionalGeneration,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        yes_tokens: list[str] = DEFAULT_YES_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_template = prompt_template
        self.yes_tokens = yes_tokens
        # TODO: This will cause an error if one of the yes_tokens is tokenized to multiple tokens.
        self.yes_ids = (
            tokenizer(yes_tokens, return_tensors="pt", add_special_tokens=False).to(self.model.device).input_ids[:, 0]
        )
        logger.debug(f"(yes_token, yes_id): {list(zip(self.yes_tokens, self.yes_ids.tolist()))}")

    def __del__(self):
        del self.yes_ids

    def ask(self, inputs: BatchEncoding) -> torch.Tensor:
        logger.debug(f"inputs.input_ids.shape: {inputs.input_ids.shape}")
        with torch.no_grad():
            if isinstance(self.model, T5ForConditionalGeneration):
                # TODO: confirm if this is the correct way to use encoder-decoder model.
                outputs = self.model(input_ids=inputs.input_ids, decoder_input_ids=inputs.input_ids)
            else:  # AutoModelForCausalLM
                outputs = self.model(**inputs)
        logger.debug(f"outputs.logits.shape: {outputs.logits.shape}")
        logits = outputs.logits[:, -1, :]
        logger.debug(f"logits.shape: {logits.shape}")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logger.debug(f"probs.shape: {probs.shape}")
        self.__log_top_k(probs) if logger.isEnabledFor(logging.DEBUG) else None
        yes_probs = probs[:, self.yes_ids]
        logger.debug(f"yes_probs.shape: {yes_probs.shape}")
        return torch.sum(yes_probs, dim=-1)

    def get_prompt(self, datapoint: str) -> str:
        return self.prompt_template.format(datapoint=self.__truncate(self.__sanitize(datapoint)))

    def get_prompts(self, datapoints: list[str]) -> list[str]:
        return [self.get_prompt(datapoint) for datapoint in datapoints]

    def __sanitize(self, datapoint: str) -> str:
        # TODO: What type of sanitization is required for the prompt?
        return datapoint.replace("###", "")

    def __truncate(self, datapoint: str) -> str:
        # TODO: The datapoint needs to be truncated to fit the maximum context size. e.g. 8192 tokens
        # See model.config.max_position_embeddings
        logger.debug(f"len(datapoint): {len(datapoint)}")
        return datapoint

    def __log_top_k(self, probs, n=10):
        for prob in probs:
            top_k = torch.topk(prob, n, dim=-1)
            for i, (idx, prob) in enumerate(zip(top_k.indices, top_k.values)):
                logger.debug(f"{i + 1:2d}\t'{self.tokenizer.decode(idx)}'\t{prob.item():.4f}")
            logger.debug("-" * 40)
