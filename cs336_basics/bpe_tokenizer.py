import regex as re
import pickle

from collections.abc import Iterable
from collections.abc import Iterator


GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.reversed_vocab = {}
        for id, token_bytes in vocab.items():
            self.reversed_vocab[token_bytes] = id

        self.merges_by_rank = {}
        rank = 0
        for merge in merges:
            self.merges_by_rank[merge] = rank
            rank+=1

        self.special_tokens = set()
        self.special_pattern = None
        if special_tokens:
            self.special_tokens = set(special_tokens)

            special_tokens = sorted(special_tokens, key=len)[::-1]
            self.special_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
            for special_token in special_tokens:
                encoded_special_token = special_token.encode("utf-8")
                if encoded_special_token not in self.reversed_vocab:
                    idx = len(vocab)
                    self.reversed_vocab[encoded_special_token] = idx
                    self.vocab[idx] = encoded_special_token

        self.token_mappings = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = re.split(self.special_pattern, text) if self.special_pattern else [text]
        result = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append(self.reversed_vocab.get(chunk.encode("utf-8")))
            else:
                result.extend(self._encode_chunk(chunk))

        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, ids))
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string
    
    def _encode_chunk(self, text: str):
        tokens = re.finditer(GPT2_TOKENIZER_REGEX, text)
        encoded_ids = []
        for token_itr in tokens:
            token = token_itr.group()
            if token not in self.token_mappings:
                merged_bytes = self._merge_word(list(bytes([b]) for b in token.encode("utf-8")))
                self.token_mappings[token] = list(map(self.reversed_vocab.get, merged_bytes))

            encoded_ids.extend(self.token_mappings.get(token))
        return encoded_ids
    
    def _merge_word(self, word: list[bytes]):
        current = word
        while True:
            start_idx = None
            min_rank = len(self.merges_by_rank)
            for i in range(len(current)-1):
                merge_candidate = (current[i], current[i+1])
                if merge_candidate in self.merges_by_rank and self.merges_by_rank.get(merge_candidate) < min_rank:
                    min_rank = self.merges_by_rank.get(merge_candidate)
                    start_idx = i

            if start_idx == None:
                break

            merged = current[start_idx] + current[start_idx + 1]
            current = current[:start_idx] + [merged] + current[start_idx + 2 :]

        return current
