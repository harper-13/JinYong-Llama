import collections
import os
from typing import (
    Dict,
    Iterator,
    List,
    Sequence,
    cast,
)

import regex
import tiktoken
from tiktoken.load import dump_tiktoken_bpe, load_tiktoken_bpe

from src.utils import VOCAB_SIZE, read_jinyong

# The tiktoken tokenizer can handle <=400k chars.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """Converts List[int] <-> str"""

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    @classmethod
    def train(cls, data: str, vocab_size: int) -> "Tokenizer":
        """BPE train."""

        # First, add tokens for each individual byte value
        if vocab_size < 2**8:
            raise ValueError(
                "vocab_size must be at least 256, so we can encode all bytes"
            )
        ranks = {}
        for i in range(2**8):
            ranks[bytes([i])] = i

        # Splinter up our data into lists of bytes
        # data = "Hello world"
        # words = [
        #     [b'H', b'e', b'l', b'l', b'o'],
        #     [b' ', b'w', b'o', b'r', b'l', b'd']
        # ]
        words: list[list[bytes]] = [
            [bytes([b]) for b in word.encode("utf-8")]
            for word in regex.findall(Tokenizer.pat_str, data, concurrent=True)
        ]

        # Now, use our data to figure out which merges we should make
        while len(ranks) < vocab_size:
            # Find the most common pair. This will become our next token
            stats = collections.Counter()
            for piece in words:
                for pair in zip(piece[:-1], piece[1:]):
                    stats[pair] += 1

            most_common_pair = max(stats, key=lambda x: stats[x])
            token_bytes = most_common_pair[0] + most_common_pair[1]
            token = len(ranks)
            # Add the new token!
            ranks[token_bytes] = token

            # Now merge that most common pair in all the words. That is, update our training data
            # to reflect our decision to make that pair into a new token.
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == most_common_pair:
                        # We found our pair! Merge it
                        new_word.append(token_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i == len(word) - 1:
                    new_word.append(word[i])
                new_words.append(new_word)
            words = new_words

        dump_tiktoken_bpe(ranks, "./dataset/jinyong_bpe_tokenizer.model")

        return cls(ranks)

    def __init__(self, mergeable_ranks: dict[bytes, int]):
        num_base_tokens = len(mergeable_ranks)

        special_tokens = ["<|begin_of_text|>", "<|end_of_text|>", "<|pad_id|>"]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name="JinYong_BPE_Tokenizer",
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(self.special_tokens)
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = self.special_tokens["<|pad_id|>"]
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(self, s: str, *, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr))
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


def get_tokenizer(jinyong_path: str) -> Tokenizer:
    bpe_model_path = "./dataset/jinyong_bpe_tokenizer.model"
    if os.path.exists(bpe_model_path):
        mergeable_ranks = load_tiktoken_bpe(bpe_model_path)
        return Tokenizer(mergeable_ranks)

    data = read_jinyong(jinyong_path)
    tokenizer = Tokenizer.train(data, VOCAB_SIZE)
    return tokenizer


# TODO: 可以尝试自己通过其他语料库训练一个 tokenizer
# 可以尝试优化 tokenizer 的代码，通过并行加速其训练过程

if __name__ == "__main__":
    tokenizer = get_tokenizer("./data/jinyong/射雕英雄传.txt")
    print(tokenizer.encode("你好", bos=True, eos=True))
    print(tokenizer.decode(tokenizer.encode("你好", bos=True, eos=True)))

    #new
    tokenizer = get_tokenizer("./data/jinyong/鹿鼎记.txt")
    print(tokenizer.encode("你好", bos=True, eos=True))
    print(tokenizer.decode(tokenizer.encode("你好", bos=True, eos=True)))