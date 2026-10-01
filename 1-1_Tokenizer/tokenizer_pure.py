import re
from typing import Optional, Iterable
# 这两种方式导入 Iterable 均可
# from collections.abc import Iterable


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None
    ):
        """
        初始化分词器。

        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                    表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        # 1.建立双向映射
        self.vocab = vocab
        self.id_to_bytes = vocab
        self.bytes_to_id = {b: i for i, b in vocab.items()}

        # 2.建立Rank字典
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        # 3.建立special_tokens的正则表达
        if special_tokens:
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(p) for p in sorted_special)
            self.special_regex = re.compile(special_pattern)   # 这里编译的是没有捕获组的正则
            # 下面的写法编译的是有捕获组的正则
            # special_pattern = f"({special_pattern})"
            # self.special_regex= re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4.构建gpt2官方预分词正则表达式
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


    def encode(self, text: str) -> list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。

        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记（Special Tokens）”和“普通文本”。
        2. 特殊标记（如 <|endoftext|>）被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。

        参数:
            text: 需要编码的原始字符串（例如 "Hello<|end|>World"）。

        返回:
            list[int]: 编码后的整数 ID 序列。
        """
        # 一、边界情况
        if not text:
            return []

        # 二、
        # A.text不含sp_tok
        if self.special_regex is None:
            return self._encode_text_segment(text)

        # B.含sp_tok
        tokens = []
        last_pos = 0
        for match in self.special_regex.finditer(text):
            # a.处理sp_tok前的文本
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))

            # b.处理sp_tok
            special_tok = match.group()
            tokens.append(self.bytes_to_id[special_tok.encode("utf-8")])

            last_pos = match.end()

        # c.边界文本
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens


    def _encode_text_segment(self, text: str) -> list[int]:
        """
        前面的单个下划线 _ 是 Python 的命名约定，表示这是一个内部方法（私有方法），意思是：
        这是类内部使用的辅助函数，外部调用者不应该直接调用它。
        """
        ids = []

        # 1.预分词
        pre_tokens = self.gpt2_pat.findall(text)

        # 2.核心操作
        for p_tok in pre_tokens:
            # （1）将当前片段转为字节序列
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            # （2）合并
            while len(byte_parts) >= 2:
                # a.找当前的best_pair
                best_pair = None
                min_rank = float('inf')

                for i in range(len(byte_parts)-1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            best_pair = pair
                            min_rank = rank

                if best_pair is None:
                    break

                # b.执行合并操作
                new_bytes_parts = []
                i = 0
                while i < len(byte_parts):
                    if i < len(byte_parts)-1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_bytes_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_bytes_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_bytes_parts

            # （3）将合并到极限后的所有字节块转换为词表中的 ID
            for part in byte_parts:
                ids.append(self.bytes_to_id[part])

        return ids


    def decode(self, ids: list[int]) -> str:
        """
        将 ID 列表解码为原始字符串。
        """
        # 1. 根据 ID 查表找回字节块
        byte_segments = [self.id_to_bytes[i] for i in ids]

        # 2. 将所有字节块按顺序拼接成一个完整的字节流
        full_bytes = b''.join(byte_segments)

        # 3. 将字节流解码为 UTF-8 字符串。
        # 使用 errors="replace" 非常关键：因为 BPE 可能会生成不完整的字节序列
        # （例如 3 字节的中文字符只产生了一部分），此时不报错而是插入替换符（）。
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。

        参数 iterable: Iterable[str]
        表示一个可迭代对象，其中每个元素都是字符串（str）。常见的如：
            文件句柄 open("file.txt")，迭代时每次产出一行；
            生成器，每次产出一个文本块；
            列表 ["hello", "world"] 等。
        返回 Iterable[int]
        实际上返回的是一个生成器，因为函数体内有 yield。调用者可以迭代它，逐个拿到整数 ID。
        """
        for chunk in iterable:   # 遍历传入的可迭代对象，每次取出一个字符串块 chunk
            """
            yield from 是 Python 3.3+ 的语法，用于委托迭代。它等价于：

            for id_ in self.encode(chunk):
                yield id_
            也就是说，它会把 self.encode(chunk) 返回的列表中的每个 ID 逐个产出，而不是一次性产出整个列表。
            """
            yield from self.encode(chunk)   # 对当前块调用已有的 encode 方法，得到一个 list[int]，即该块编码后的 ID 列表
