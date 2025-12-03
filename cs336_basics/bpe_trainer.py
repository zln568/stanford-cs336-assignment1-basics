import regex as re
import multiprocessing as mp
import os
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.common import FIXTURES_PATH
from pathlib import Path


GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # vocabulary initialization
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    i = 256
    for special_token in special_tokens:
        vocab[i] = special_token.encode("utf-8")
        i+=1

    # pre-tokenization
    pretoken_count = pretokenize(input_path, special_tokens)

    # compute bpe merges
    return compute_bpe_merges(pretoken_count, vocab, vocab_size)

def pretokenize(
        input_path: str | os.PathLike,
        special_tokens: list[str]
        ):
    special_pattern = re.compile("|".join(re.escape(token) for token in special_tokens)) if len(special_tokens) > 0 else None

    chunk_results = []
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            
            chunk_str = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_results.append(pool.apply_async(pretokenize_chunk, (chunk_str, special_pattern)))

    pool.close()
    pool.join()

    pretoken_count_map = {}
    for chunk_result in chunk_results:
        chunk_pretoken_count_map = chunk_result.get()
        for key in chunk_pretoken_count_map:
            pretoken_count_map[key] = pretoken_count_map.get(key, 0) + chunk_pretoken_count_map.get(key)
    
    return pretoken_count_map

def pretokenize_chunk(
        chunk_str: str,
        special_pattern: re.Pattern
        ):
    pretoken_count = {}
    sub_chunks = special_pattern.split(chunk_str) if special_pattern else [chunk_str]

    for chunk in sub_chunks:
        tokens = re.finditer(GPT2_TOKENIZER_REGEX, chunk)
        for token in tokens:
            indices = tuple(map(int, token.group().encode("utf-8")))
            pretoken_count[indices] = pretoken_count.get(indices,0) + 1

    return pretoken_count
           

def compute_bpe_merges(
          pretoken_count: dict[tuple, int], 
          vocab: dict[int, bytes],
          vocab_size_limit: int):
    pair_counts = {}
    pretoken_mapping = {}
    for indices in pretoken_count:
        pretoken_mapping[indices] = indices
        token_count = pretoken_count.get(indices)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            pair_counts[(index1, index2)] = pair_counts.get((index1, index2),0) + token_count

    pair_counts_pq = []
    for pair, count in pair_counts.items():
        index1, index2 = pair
        heapq.heappush(pair_counts_pq, (-count, BytePair((vocab.get(index1), vocab.get(index2))), pair))
    
    vocab_idx = len(vocab)
    merges = []
    while len(vocab) < vocab_size_limit:
        max_pair = None
        while len(pair_counts_pq) > 0:
            negated_max_count, byte_pair, max_pair = heapq.heappop(pair_counts_pq)
            if -negated_max_count == pair_counts.get(max_pair):
                break
       
        if max_pair == None:
            break

        index1, index2 = max_pair

        vocab[vocab_idx] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        changed_pairs = set()
        changed_pairs.add(max_pair)
        for indices in pretoken_count:
            token_count = pretoken_count.get(indices)
            merged_indices = merge_and_update(pretoken_mapping.get(indices), max_pair, vocab_idx, token_count, pair_counts, changed_pairs)
            pretoken_mapping[indices] = merged_indices

        for pair in changed_pairs:
            index1, index2 = pair
            count = pair_counts.get(pair)
            if count > 0:
                heapq.heappush(pair_counts_pq, (-count, BytePair((vocab.get(index1), vocab.get(index2))), pair))

        vocab_idx+=1
    
    return vocab, merges


def merge_and_update(
        indices: list[int], 
        pair: tuple[int, int], 
        vocab_idx: int,
        token_count: int,
        pair_counts: dict[tuple, int],
        changed_pairs: set[tuple]
        ) -> list[int]:    
    merged_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            if len(merged_indices) > 0:
                changed_pair = (merged_indices[-1], vocab_idx)
                pair_counts[changed_pair] = pair_counts.get(changed_pair,0) + token_count
                changed_pairs.add(changed_pair)

            pair_counts[(indices[i], indices[i+1])] -= token_count
            if i > 0:
                changed_pair = (indices[i-1], indices[i])
                pair_counts[changed_pair] -= token_count
                changed_pairs.add(changed_pair)
            merged_indices.append(vocab_idx)

            i += 2
        else:
            if len(merged_indices) > 0 and merged_indices[-1] == vocab_idx:
                changed_pair = (vocab_idx, indices[i])
                pair_counts[changed_pair] = pair_counts.get(changed_pair,0) + token_count
                changed_pairs.add(changed_pair)
                if i > 0:
                    changed_pair = (indices[i-1], indices[i])
                    pair_counts[changed_pair] -= token_count
                    changed_pairs.add(changed_pair)
            merged_indices.append(indices[i])
            i += 1
    return tuple(merged_indices)

class BytePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "BytePair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: "BytePair") -> bool:
        return self.pair == other


if __name__ == "__main__":
    train_bpe(
        input_path=FIXTURES_PATH / "corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"])
    