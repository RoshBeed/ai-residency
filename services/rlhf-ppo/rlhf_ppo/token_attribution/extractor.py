"""Tokenising generated text back into tokens and ids."""

from typing import Any

__all__ = ["TokenExtractor"]


class TokenExtractor:
    """Recovers the token sequence behind a generated string."""

    def __init__(self, tokenizer: Any) -> None:
        """Store the tokenizer to use.

        Args:
            tokenizer: A Hugging Face tokenizer.
        """
        self.tokenizer = tokenizer

    def extract_tokens(self, generated_text: str) -> tuple[list[str], list[int]]:
        """Return the tokens and token ids of a generated string.

        Args:
            generated_text: Text produced by the policy model.

        Returns:
            The token strings and their corresponding vocabulary ids.
        """
        tokens: list[str] = self.tokenizer.tokenize(generated_text)
        token_ids: list[int] = self.tokenizer.encode(generated_text, add_special_tokens=False)
        return tokens, token_ids
