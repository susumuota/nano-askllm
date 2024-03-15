# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

logger = logging.getLogger(__name__)


class AskLLM:
    # A prompt template. See https://arxiv.org/abs/2402.09668 page 4.
    DEFAULT_PROMPT_TEMPLATE_PREFIX = "###\n"
    DEFAULT_PROMPT_TEMPLATE_POSTFIX = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS:
- yes
- no
"""  # noqa: E501

    # Each of the word must be tokenized to a single token.
    DEFAULT_YES_TOKENS = ["yes", "Yes"]

    # The number of tokens to avoid to exceed the maximum context size.
    # TODO: 1?
    TOKEN_MARGIN = 2

    def __init__(
        self,
        tokenizer: AutoTokenizer | T5Tokenizer,
        model: AutoModelForCausalLM | T5ForConditionalGeneration,
        prompt_template_prefix: str = DEFAULT_PROMPT_TEMPLATE_PREFIX,
        prompt_template_postfix: str = DEFAULT_PROMPT_TEMPLATE_POSTFIX,
        yes_tokens: list[str] = DEFAULT_YES_TOKENS,
        max_tokens: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_template_prefix = prompt_template_prefix
        self.prompt_template_postfix = prompt_template_postfix
        self.yes_tokens = yes_tokens
        # Convert yes_tokens to yes_ids.
        if not all([len(self.tokenizer.encode(yes, add_special_tokens=False)) == 1 for yes in self.yes_tokens]):
            raise ValueError("Each of the word must be tokenized to a single token.")
        self.yes_ids: torch.Tensor = (
            self.tokenizer(yes_tokens, return_tensors="pt", add_special_tokens=False)
            .to(self.model.device)
            .input_ids[:, 0]
        )
        logger.debug(f"(yes_token, yes_id): {list(zip(self.yes_tokens, self.yes_ids.tolist()))}")
        # Set the maximum number of tokens to limit the context size.
        self.max_tokens: int = (
            max_tokens
            if max_tokens is not None
            else (self.model.config.n_positions if self.__is_t5() else self.model.config.max_position_embeddings)
        )
        logger.debug(f"max_tokens: {self.max_tokens}")
        prompt_len = sum(
            [self.__token_length(item) for item in [self.prompt_template_prefix, self.prompt_template_postfix]]
        )
        self.max_datapoint_tokens: int = self.max_tokens - self.TOKEN_MARGIN - prompt_len
        logger.debug(f"max_datapoint_tokens: {self.max_datapoint_tokens}")

    def __del__(self):
        del self.yes_ids

    def ask(self, datapoints: list[str]) -> torch.Tensor:
        prompts = self.get_prompts(datapoints)
        if self.__is_t5():
            # TODO: confirm that this is the correct way to truncate the input.
            inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length").to(self.model.device)
        else:  # AutoTokenizer
            # TODO: truncation
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        logger.debug(f"inputs.input_ids.shape: {inputs.input_ids.shape}")
        with torch.no_grad():
            if self.__is_t5():
                # TODO: Confirm that this is the correct way to forward on the encoder-decoder model.
                outputs = self.model(input_ids=inputs.input_ids, decoder_input_ids=inputs.input_ids)
            else:  # AutoModelForCausalLM
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        logger.debug(f"outputs.logits.shape: {outputs.logits.shape}")
        logits = outputs.logits[:, -1, :]
        logger.debug(f"logits.shape: {logits.shape}")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logger.debug(f"probs.shape: {probs.shape}")
        self.__log_top_k(probs) if logger.isEnabledFor(logging.DEBUG) else None
        yes_probs = probs[:, self.yes_ids]
        logger.debug(f"yes_probs.shape: {yes_probs.shape}")
        del inputs, outputs, logits, probs
        return torch.sum(yes_probs, dim=-1)

    def get_prompt(self, datapoint: str) -> str:
        # TODO: Cache prompt template to avoid redundant tokenization.
        truncated = self.__truncate(self.__sanitize(datapoint))
        prompt = self.prompt_template_prefix + truncated + self.prompt_template_postfix
        return prompt

    def get_prompts(self, datapoints: list[str]) -> list[str]:
        return [self.get_prompt(datapoint) for datapoint in datapoints]

    def __is_t5(self) -> bool:
        return isinstance(self.model, T5ForConditionalGeneration)

    def __sanitize(self, datapoint: str) -> str:
        # TODO: What type of sanitization is required for the prompt?
        return datapoint.replace("###", "")

    def __truncate(self, datapoint: str) -> str:
        # The datapoint needs to be truncated to fit the maximum context size.
        # TODO: Reuse tokens to avoid redundant tokenization.
        logger.debug(f"len(datapoint): {len(datapoint)}")
        tokens = self.__token_encode(datapoint)
        tokens_len = len(tokens)
        logger.debug(f"len(tokens): {tokens_len}")
        if tokens_len > self.max_datapoint_tokens:
            truncated_tokens = tokens[: self.max_datapoint_tokens]
            logger.warning(
                f"Truncated the datapoint to fit the maximum context size. {tokens_len} > {self.max_datapoint_tokens}"
            )
            logger.debug(f"len(truncated_tokens): {len(truncated_tokens)}")
            # TODO: Fix bug that flan-t5 cannot recover the original text from the truncated tokens.
            truncated_datapoint = self.__token_decode(truncated_tokens)
            logger.debug(f"len(truncated_datapoint): {len(truncated_datapoint)}")
            return truncated_datapoint
        return datapoint

    def __token_encode(self, datapoint: str) -> list[int]:
        return self.tokenizer.encode(datapoint, add_special_tokens=False)

    def __token_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __token_length(self, datapoint: str) -> int:
        return len(self.__token_encode(datapoint))

    def __log_top_k(self, probs: torch.Tensor, k: int = 10):
        for prob in probs:
            top_k = torch.topk(prob, k, dim=-1)
            for i, (idx, prob) in enumerate(zip(top_k.indices, top_k.values)):
                logger.debug(f"{i + 1:2d}\t'{self.tokenizer.decode(idx)}'\t{prob.item():.4f}")
