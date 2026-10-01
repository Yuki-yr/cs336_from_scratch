import regex as re
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
        # 1. 建立双向映射，方便查表
        self.vocab = vocab   # ID -> 字节块
        self.id_to_bytes = vocab
        self.bytes_to_id = {b: i for i, b in vocab.items()}   # 字节块 -> ID

        # 2. 将合并规则转换为Rank字典
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则。
        # 字典结构为: {(byte_a, byte_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        # special_tokens or [] 利用了 Python 中 or 的短路求值特性，它会把所有假值都替换成 []，包括空列表、空字符串、0 等
        # 等价于 self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens = special_tokens or []

        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:
            # 关键：必须按照长度从长到短排序（reverse=True）。
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(p) for p in sorted_special)
            # re.compile 将正则字符串编译成一个 Pattern 对象，缓存起来，后续多次使用不必重复解析，提高效率。
            # 编译后的对象可以调用 .split()、.findall()、.search() 等方法。
            self.special_regex = re.compile(special_pattern)   # 这里编译的是没有捕获组的正则
            # 下面的写法编译的是有捕获组的正则
            # special_pattern = f"({special_pattern})"
            # self.special_regex= re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4. GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
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
        # --- 步骤 1: 边界情况检查 ---
        # 如果输入是空字符串或 None，直接返回空列表。
        # 这是为了防止后续逻辑在处理空文本时产生错误。
        if not text:
            return []

        # --- 步骤 2: 情况 A - 快速路径 (Fast Path) ---
        # 如果我们在初始化时没有定义任何特殊标记（或者特殊标记列表为空），
        # 那么整个文本都可以被视为一段连续的“普通文本”。
        # 我们直接调用内部方法 _encode_text_segment 进行 BPE 处理并返回结果。
        if not self.special_regex:
            return self._encode_text_segment(text)

        # --- 步骤 3: 情况 B - 处理含有特殊标记的复杂文本 ---
        # 此时文本中可能混有普通文字和特殊标记，我们需要像“剪刀”一样把它们切开。
        tokens = []
        last_pos = 0   # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。

        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项。
        # finditer 的好处是它提供了 match.start() 和 match.end()，
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束。
        for match in self.special_regex.finditer(text):
            # 3.1 提取并处理“前置普通文本”
            # 这里的区间是 [last_pos, match.start())。
            # " hello <|endoftext|> world"
            # 这段文本是夹在两个特殊标记之间（或者开头到第一个特殊标记之间）的普通文字。
            pre_text = text[last_pos:match.start()]

            # 如果这两个标记之间确实有文字（长度 > 0）
            if pre_text:
                # 调用核心 BPE 逻辑。_encode_text_segment 会执行：
                # 1. GPT-2 预分词正则切分。
                # 2. 字节化。
                # 3. 按照 merges 规则进行贪婪合并。
                tokens.extend(self._encode_text_segment(pre_text))
                # pre_tokens : [1,2,3,...] self._encode_text_segment: [4,5,6] tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]

            # 3.2 处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串（如 "<|endoftext|>"）。
            special_tok = match.group()
            # 核心原则：特殊标记不参与 BPE 合并！
            # 我们直接将其编码为 UTF-8 字节，然后在词表中查找其 ID。
            # 注意：这些标记在 train_bpe 阶段必须已经被手动加入到了词表中。
            tokens.append(self.bytes_to_id[special_tok.encode("utf-8")])

            # 3.3 更新游标
            # 将游标移动到当前匹配项的末尾，为寻找下一个片段做准备。
            last_pos = match.end()

        # --- 步骤 4: 处理“收尾文本” ---
        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]
        if remaining_text:
            # 剩余部分同样作为普通文本片段进行 BPE 编码。
            tokens.extend(self._encode_text_segment(remaining_text))

        # 返回拼接好的所有 ID 列表
        return tokens



    def _encode_text_segment(self, text: str) -> list[int]:
        """
        前面的单个下划线 _ 是 Python 的命名约定，表示这是一个内部方法（私有方法），意思是：
        这是类内部使用的辅助函数，外部调用者不应该直接调用它。
        """
        ids = []   # ids 是 identifiers 的缩写，意思是“标识符”或“编号”
        # 1. 使用GPT-2正则进行预分词，将文本拆成单词/标点符号块
        # 例如："Hello world!" -> ["Hello", " world", "!"]
        pre_tokens = self.gpt2_pat.findall(text)

        # 2.核心操作
        for p_tok in pre_tokens:
            # 第一步：将当前片段转为字节序列，并将每个字节看作一个独立的“部分（Part）”
            # 例如："Hello" -> [b'H', b'e', b'l', b'l', b'o']
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            # 第二步：反复执行合并，直到没有符合条件的合并规则为止
            """
            注意：
            编码时必须按 rank 顺序合并，才能得到和训练时一致的分词结果。
            否则得到的字节块和训练时不一致，查词表就会出错。
            """
            while len(byte_parts) >= 2:
                # a.找当前的best_pair
                best_pair = None
                min_rank = float('inf')

                for i in range(len(byte_parts)-1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                # 如果找不到任何可以合并的规则，退出当前片段的合并过程
                if best_pair is None:
                    break

                # b.执行合并操作。
                # 遍历当前序列，将所有出现的 best_pair 替换成合并后的长字节块。
                new_byte_parts = []
                i = 0
                # [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                while i < len(byte_parts):
                    # 如果当前两个部分匹配最高优规则
                    if i < len(byte_parts)-1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts

            # 第三步：将合并到极限后的所有字节块转换为词表中的 ID
            for part in byte_parts:
                ids.append(self.bytes_to_id[part])

        return ids


    """
    完整示例
    vocab = {
        ..., 
        256: b'ab',
        258: b'c',
        300: b'\xe4\xbd\xa0',   # "你"
        301: b'\xe5\xa5\xbd',   # "好"
    }
    ids = [256, 258, 300, 301]
    
    解码过程：
    byte_segments = [b'ab', b'c', b'\xe4\xbd\xa0', b'\xe5\xa5\xbd']
    full_bytes = b'abc\xe4\xbd\xa0\xe5\xa5\xbd'
    result = full_bytes.decode("utf-8", errors="replace")
    # result = "abc你好"
    """
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
        """
        为什么是“内存高效”的？
            调用 encode_iterable 时，不会立即执行函数体，而是返回一个生成器对象。
            只有当你开始迭代这个生成器时，才会逐块处理：
                从 iterable 取一块文本；
                编码成 ID 列表；
                逐个 yield 出去；
                处理下一块。
            整个过程不需要一次性把所有文本读入内存，也不需要一次性生成所有 ID。
            对于大文件（如几 GB 的文本），可以边读边编码，内存占用主要取决于单个块的大小。
        """

