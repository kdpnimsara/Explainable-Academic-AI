from transformers import AutoTokenizer


class HuggingFaceTokenizerAdapter:
    """Adapt Hugging Face tokenizers to the project interface."""

    def __init__(self, tokenizer):
        """Initialize with an injected Hugging Face tokenizer."""

        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("Tokenizer must provide encode().")

        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("Tokenizer must provide decode().")

        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "A fast tokenizer is required for offset mapping."
            )

        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        local_files_only: bool = False,
    ):
        """Load a pretrained fast tokenizer."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
        )

        return cls(tokenizer)

    @property
    def model_max_length(self) -> int:
        """Return the tokenizer's configured maximum length."""

        return self.tokenizer.model_max_length

    @property
    def vocabulary_size(self) -> int:
        """Return the tokenizer vocabulary size."""

        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs without special tokens."""

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs for inspection."""

        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list.")

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Return token IDs and their original character positions."""

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if len(token_ids) != len(offsets):
            raise ValueError(
                "Token IDs and offset mappings have different lengths."
            )

        return (
            list(token_ids),
            [tuple(offset) for offset in offsets],
        )