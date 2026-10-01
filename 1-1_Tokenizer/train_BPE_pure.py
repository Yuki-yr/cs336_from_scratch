import os
from typing import Union
import regex as re
from collections import Counter, defaultdict

def train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: list[str],
):
    # --- 1、构建基础词表 ---
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)

    # --- 2、读取语料，并以特殊 Token 进行分割
    # (1) 读取语料
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # (2) 以特殊 Token 进行分割
    if special_tokens:   # 表示 special_tokens 这个列表非空
        # 第一步：构造匹配特殊 Token 的正则
        special_regex = "|".join(re.escape(p) for p in special_tokens)
        # 第二步：切分文本
        pattern = f"({special_regex})"   # 加上捕获组，用于re.split
        parts = re.split(pattern, text)
        # 第三步：过滤掉特殊 Token
        train_segments = [p for p in parts if p not in special_tokens]

    else:
        train_segments = [text]

    # --- 3、预分词（Pre-tokenization）并统计词频 ---
    # (1) 预分词
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # (2) 统计词频
    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    # --- 构建高效数据结构以支持快速合并 ---
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    stats = defaultdict(int)
    indices = defaultdict(set)
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)

    merges = []


    # --- 4. 迭代合并流程 ---
    for _ in range(num_merges):
        if not stats:
            break

        # --- 4a. 寻找最佳 Pair ---
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        if stats[best_pair] <= 0:   # 它是“逻辑上冗余、工程上合理”的防御性检查。
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # --- 4b. 获取需要更新的单词(索引) ---
        relevant_indices = list(indices[best_pair])

        # --- 4c. 遍历并更新所有受影响的单词、统计信息和倒排索引 ---
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]

            """
            注意（易错）:
            不能用 for i in range(len(word)-1), 因为range 只求值一次，
            range(len(word)-1) 在进入循环时长度就被冻结了，
            一旦内部执行 del word[i+1]，len(word) 变小，但 i 仍会继续加，
            从而出现 IndexError
            """
            i = 0
            while i < len(word)-1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    # --- 匹配到 `best_pair`，执行合并 ---

                    # 1. 更新旧的邻居 Pair 的频率：
                    if i > 0:
                        pre_pair = (word[i-1], word[i])
                        stats[pre_pair] -= freq
                        if stats[pre_pair] == 0:
                            del stats[pre_pair]

                    if i < len(word)-2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 2. 修改单词结构：将 (word[i], word[i+1]) 替换为 new_token
                    word[i] = new_token
                    del word[i+1]

                    # 3. 添加新产生的邻居 Pair 的频率和索引
                    #    - 新的左邻居：(word[i-1], new_token)
                    if i > 0:
                        new_prev = (word[i-1], new_token)
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)

                    #    - 新的右邻居：(new_token, word[i+1]) (注意：word[i+1] 是旧的 word[i+2])
                    if i < len(word) - 1:
                        new_next = (new_token, word[i + 1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                else:
                    i += 1

        # 4d. 清理：移除已完全合并的 `best_pair`
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]


    # --- 5. 构建最终的词表 ---
    # 添加 BPE 合并产生的 Token，ID 从 256 开始，按合并顺序递增
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]

    # 添加特殊 Token
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        new_id = len(vocab)
        vocab[new_id] = s_bytes

    return vocab, merges

