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
        if not all([self.__token_length(yes) == 1 for yes in self.yes_tokens]):
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
            else (
                self.model.config.n_positions  # TODO: 512 or no truncation needed for T5 model? see https://github.com/huggingface/transformers/issues/8047  # noqa: E501
                if isinstance(self.model, T5ForConditionalGeneration)
                else self.model.config.max_position_embeddings
            )
        )
        logger.debug(f"max_tokens: {self.max_tokens}")
        prompt_len = sum(
            [self.__token_length(item) for item in [self.prompt_template_prefix, self.prompt_template_postfix]]
        )
        self.max_datapoint_tokens: int = self.max_tokens - prompt_len - 1  # -1 for special token
        logger.debug(f"max_datapoint_tokens: {self.max_datapoint_tokens}")

    def __del__(self):
        del self.yes_ids

    def ask(self, datapoints: list[str]) -> torch.Tensor:
        prompts = self.get_prompts(datapoints)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        logger.debug(f"inputs.input_ids.shape: {inputs.input_ids.shape}")  # (batch_size, seq_len)
        # obtain logits by calling `generate` with output_logits=True and return_dict_in_generate=True
        # it should be same logits by calling `forward`. e.g.,
        # with torch.no_grad():
        #     logits = model(input_ids).logits[:, -1, :]
        # see https://github.com/huggingface/transformers/blob/56baa03380fc17d85705240ebbc57c075a9b3f23/tests/generation/test_utils.py#L3479  # noqa: E501
        outputs = self.model.generate(**inputs, max_new_tokens=1, output_logits=True, return_dict_in_generate=True)
        logits = outputs.logits[0]
        logger.debug(f"logits.shape: {logits.shape}")  # (batch_size, vocab_size)
        # convert logits to probabilities
        # see https://github.com/huggingface/transformers/blob/56baa03380fc17d85705240ebbc57c075a9b3f23/tests/generation/test_utils.py#L3507  # noqa: E501
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logger.debug(f"probs.shape: {probs.shape}")  # (batch_size, vocab_size)
        self.__log_topk(probs) if logger.isEnabledFor(logging.DEBUG) else None
        yes_probs = probs[:, self.yes_ids]
        logger.debug(f"yes_probs.shape: {yes_probs.shape}")  # (batch_size, num_yes_tokens)
        scores = torch.sum(yes_probs, dim=-1)
        logger.debug(f"scores.shape: {scores.shape}")  # (batch_size,)
        del yes_probs, probs, logits, outputs, inputs
        return scores

    def get_prompt(self, datapoint: str) -> str:
        # TODO: Cache prompt template to avoid redundant tokenization.
        truncated = self.__truncate(self.__sanitize(datapoint))
        prompt = self.prompt_template_prefix + truncated + self.prompt_template_postfix
        return prompt

    def get_prompts(self, datapoints: list[str]) -> list[str]:
        return [self.get_prompt(datapoint) for datapoint in datapoints]

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
            # TODO: flan-t5 cannot recover the original text
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

    def __log_topk(self, probs: torch.Tensor, k: int = 10) -> None:
        for i, prob in enumerate(probs):
            tops = torch.topk(prob, k, dim=-1)
            logger.debug("-" * 23)
            logger.debug(f"datapoint {i + 1}")
            for j, (idx, val) in enumerate(zip(tops.indices, tops.values)):
                logger.debug(f"{j + 1:2d} | {self.tokenizer.decode(idx):8s} | {val.item():.4f}")
        logger.debug("-" * 23)
