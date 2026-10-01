import json
import os
from typing import Union
import regex as re
from collections import Counter, defaultdict


def train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: list[str],
):
    # --- 1. 初始化基础词表 ---
    # 词表从 0 到 255 的字节开始，这是BPE的基础单位
    """
    注意：
    bytes() 构造函数对整数和可迭代对象的处理方式完全不同：
    如果参数是整数，bytes(n) 会创建一个长度为 n 的、全部填 0 的字节串。
    如果参数是可迭代对象（比如列表、元组），bytes(iterable) 会把可迭代对象里的每个整数当作一个字节的值，依次组成字节串。

    For example:
    当 i = 65 时：
    python
    bytes([65])   # b'A'
    bytes((65,))  # b'A'，元组也可以
    bytes(65)     # b'\x00' * 65，长度为 65 的全零字节串
    """
    vocab = {i: bytes([i]) for i in range(256)}

    # 计算需要进行的合并次数
    # 目标词表大小 = 基础字节数 (256) + 特殊 Token 数 + 需要新生成的 Token 数。
    num_merges = vocab_size - 256 - len(special_tokens)


    # --- 2. 读取语料，并按特殊 Token 分割 ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 如果指定了特殊 Token，我们需要在开始统计之前将它们从语料中“隔离”出来。
    # 这能防止 BPE 规则将特殊 Token（如 <|endoftext|>）拆开或与普通文本混合。
    """
    special_tokens = ["<|endoftext|>", "<pad>"]
    text = "Hello World World<|endoftext|>Hello happy happy<|endoftext|>!<pad>Good"
    """
    if special_tokens:
        # 第一步：构造匹配特殊 Token 的正则
        """
        re 是 Python 标准库里的 正则表达式模块，全称 regular expression
        
        re.escape()会把正则里有特殊含义的字符转义
        e.g: 
        re.escape("<|endoftext|>") 这里 | 是正则的“或”，必须转义成 \|。
        所以结果大致是：<\|endoftext\|>
        """
        special_regex = "|".join(re.escape(t) for t in special_tokens)   # 最终得到类似：special_regex = r"<\|endoftext\|>|<pad>"
        # 第二步：切分文本
        # 加上捕获组，用于re.split
        """
        加上捕获组，用于 re.split
        外面的圆括号 (...) 是捕获组。
        如果没有这个括号，re.split 会把分隔符丢掉；有了它，分隔符本身会保留在结果列表里。
        """
        pattern = f"({special_regex})"   # 相当于"(" + special_regex + ")"
        parts = re.split(pattern, text)
        """
        经此，
        parts = [
            "Hello World World",
            "<|endoftext|>",
            "Hello happy happy",
            "<|endoftext|>",
            "!",
            "<pad>",
            "Good"
        ]
        """
        # 第三步：过滤掉特殊 Token
        train_segments = [p for p in parts if p not in special_tokens]
        """
        经此，
        train_segments = [
            "Hello World World",
            "Hello happy happy",
            "!",
            "Good"
        ]
        """
    else:
        train_segments = [text]
        """
        先保留special tokens再进行过滤的原因：
        防止train_segments中出现空字符串。
        与如下规则有关：
        如果分隔符出现在字符串开头，第一个元素就是空字符串。
        如果分隔符出现在字符串结尾，最后一个元素就是空字符串。
        如果两个分隔符相邻，它们中间会有一个空字符串。
        如果分隔符前后都有普通内容，就不会产生空字符串。
        """


    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    # (1) 预分词
    # 使用 GPT-2 的 BPE 预分词正则表达式。
    # GPT-2 正则表达式的作用是执行“预分词（Pre-tokenization）”。 它的规则是：
    # (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    # (2)保护空格：它通常会把单词前面的空格和单词连在一起，作为一个整体。
    # text = "Hello World test! ..."
    # 分割后 words = ['Hello', ' World', ' test', '!', ' ...']
    """
    它模仿的是 GPT-2 的 BPE 预分词规则, 不必深究。
    """
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # (2) 统计词频
    """
    Counter 是 collections 模块里的一个字典子类，专门用来计数。
    这里 raw_counts 用来存储：
        键：一个“单词”的字节元组，例如 (b'h', b'e', b'l', b'l', b'o')
        值：这个单词在语料中出现的次数
    """
    raw_counts = Counter()   # 词频统计容器
    """
    继上例：
        train_segments = [
            "Hello World World",
            "Hello happy happy",
            "!",
            "Good"
        ]
    """
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        """
        e.g:
        经此，
        words = ['Hello', ' World', ' World']
        """
        for word in words:
            """
            这一行是核心，分几步理解：
            第一步：word.encode("utf-8")
            把字符串 word 编码成 UTF-8 字节序列。
            e.g:
            word = "Hi"
            word.encode("utf-8")  # 得到 b'Hi'
            
            第二步：for b in word.encode("utf-8")
            遍历这个 bytes 对象时，得到的不是字符，而是整数。
            e.g:
                for b in b'Hi':
                    print(b)
                输出：
                    72
                    105
                    
            第三步：bytes([b])
            把整数 b 转回一个单字节的 bytes 对象。
            e.g:
                bytes([72])   # b'H'
                bytes([105])  # b'i'
            
            第四步：生成元组
                tuple(bytes([b]) for b in word.encode("utf-8"))
                这是一个生成器表达式，把每个单字节 bytes 对象收集成一个元组。
                例如 "Hi"：
                (b'H', b'i')
                
            第五步：作为 Counter 的键
            Counter 的键必须是不可变的。
            列表 list 是可变的，不能做键；元组 tuple 是不可变的，可以做键。
            所以这里把字节序列转成元组
            """
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
            """
            e.g:
            假设 train_segments 中有一个片段：
                segment = "Hello World test!"
            预分词后：
                words = ['Hello', ' World', ' test', '!']
            逐个处理：
                "Hello" → UTF-8 字节：b'Hello' → 元组 (b'H', b'e', b'l', b'l', b'o')
                " World" → b' World' → (b' ', b'W', b'o', b'r', b'l', b'd')
                " test" → b' test' → (b' ', b't', b'e', b's', b't')
                "!" → b'!' → (b'!',)
            最终 raw_counts 可能变成：
                {
                    (b'H', b'e', b'l', b'l', b'o'): 1,
                    (b' ', b'W', b'o', b'r', b'l', b'd'): 1,
                    (b' ', b't', b'e', b's', b't'): 1,
                    (b'!',): 1
                }
            如果语料中 "Hello" 出现了多次，对应的值就会增加。
            """

    # --- 构建高效数据结构以支持快速合并 ---
    # words_list: 存储每个单词的字节列表。使用 list 而不是 tuple，因为 BPE 合并会修改单词内部结构。
    # counts_list: 存储对应单词的频率。
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))   # 转换为 list 以便后面修改
        counts_list.append(freq)
    """
    经此，
    words_list = [
        [b'H', b'e', b'l', b'l', b'o'],          # 下标 0
        [b' ', b'W', b'o', b'r', b'l', b'd'],    # 下标 1
        [b' ', b't', b'e', b's', b't'],          # 下标 2
        [b'!']                                    # 下标 3
    ]

    counts_list = [1, 1, 1, 1]
    """
    # defaultdict(int) 是一个“带默认初始值”的字典。当你访问一个字典中不存在的键时，它不会报错，而是自动为这个键创建一个默认值 0，而在使用普通字典进行计数时，你必须先检查键是否存在，否则会触发 KeyError。
    # stats: 存储所有可能的相邻字节对 (pair) 及其全局出现频率。
    # 结构：{(byte_a, byte_b): frequency}
    stats = defaultdict(int)
    # indices: 倒排索引。存储 pair -> {包含该 pair 的单词在 words_list 中的下标集合}
    # 这个结构是性能优化的关键，用于快速找到需要更新的单词。
    indices = defaultdict(set)

    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
    """
    经此，
    最终数据结构：
    stats = {
        (b'H', b'e'): 1,
        (b'e', b'l'): 1,
        (b'l', b'l'): 1,
        (b'l', b'o'): 1,
        (b' ', b'W'): 1,
        (b'W', b'o'): 1,
        (b'o', b'r'): 1,
        (b'r', b'l'): 1,
        (b'l', b'd'): 1,
        (b' ', b't'): 1,
        (b't', b'e'): 1,
        (b'e', b's'): 1,
        (b's', b't'): 1
    }

    indices = {
        (b'H', b'e'): {0},
        (b'e', b'l'): {0},
        (b'l', b'l'): {0},
        (b'l', b'o'): {0},
        (b' ', b'W'): {1},
        (b'W', b'o'): {1},
        (b'o', b'r'): {1},
        (b'r', b'l'): {1},
        (b'l', b'd'): {1},
        (b' ', b't'): {2},
        (b't', b'e'): {2},
        (b'e', b's'): {2},
        (b's', b't'): {2}
    }
    """
    merges = []

    """
    补充语法知识
    lambda 是 Python 中创建匿名函数的关键字。
    它的核心特点是：没有函数名、只能写一个表达式、自动返回表达式的结果。
    
    1、基本语法
    lambda 参数1, 参数2, ...: 表达式
    
    lambda 是关键字。
    冒号左边是参数列表，可以没有参数，也可以有多个参数。
    冒号右边是一个表达式，它的值就是函数的返回值。
    不需要写 return，表达式的结果自动返回。
    
    例如：
    f = lambda x: x + 1
    print(f(3))  # 4
    等价于：
    def f(x):
        return x + 1
    
    2、 常见用法之一：作为高阶函数的参数
    e.g: max / min 的 key
    本代码中的用法：
    best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        stats.items() 产生 (键, 值) 元组，例如 ((b'e', b'l'), 5)。
        key=lambda x: (x[1], x[0]) 告诉 max：对每个元素 x，用 (x[1], x[0]) 作为比较依据。
        x[1] 是频率，x[0] 是相邻对。
        所以比较规则是：先比频率，频率相同再比相邻对。
    等价于：
        def get_key(x):
            return (x[1], x[0])
        best_pair = max(stats.items(), key=get_key)[0]
    """
    # --- 4. 迭代合并流程 ---
    """
    具体例子，见：
    "F:\LLM\notebooks\BPE算法迭代合并流程具体举例.md"
    """
    # 循环执行 `num_merges` 次，每次找到并应用一个最佳合并规则
    for _ in range(num_merges):
        # 边界情况 1: 如果 `stats` 为空（所有可能的对都已合并或频率为0），则停止
        """
        注意：not stats 只看键，不看值
        not stats 判断的是字典里有没有键，而不是键对应的值是不是 0。
        """
        if not stats:
            break

        # --- 4a. 寻找最佳 Pair ---
        # 目标：找到当前 `stats` 中频率最高、且字典序最大的 Pair
        """
        易错点：
        max(stats, ...) 中的 stats 是一个 字典。对字典做 max 时，遍历的是它的键，不是 (键, 值) 对
        stats.items() 产生 (pair, freq) 对
        不要把 stats.items() 错写为 stats
        """
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        # 边界情况 2: 如果最佳 Pair 的频率已经降到 0（可能是在之前的迭代中由于其组成部分被合并了），则停止
        if stats[best_pair] <= 0:   # 它是“逻辑上冗余、工程上合理”的防御性检查。
            break

        # 记录这次合并
        merges.append(best_pair)
        # 创建新的 Token（合并后的字节序列）
        new_token = best_pair[0] + best_pair[1]

        # --- 4b. 获取需要更新的单词(索引) ---
        # 使用倒排索引 `indices`，快速获取所有包含 `best_pair` 的单词的下标
        # 必须复制一份 `relevant_indices`，因为后面的循环会修改 `indices` 和 `stats`
        relevant_indices = list(indices[best_pair])

        # --- 4c. 遍历并更新所有受影响的单词、统计信息和倒排索引 ---
        for idx in relevant_indices:
            word = words_list[idx]   # 获取单词
            freq = counts_list[idx]  # 获取单词的频率

            # 扫描当前单词，找到所有 `best_pair` 的出现位置
            i = 0
            while i < len(word) - 1:
                # 检查当前位置 `i` 和 `i+1` 是否匹配 `best_pair`
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    # --- 匹配到 `best_pair`，执行合并 ---

                    # 1. 更新旧的邻居 Pair 的频率：
                    #    - 左邻居：(word[i-1], word[i])
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
                        new_prev = (word[i-1], new_token)   # word[i] 现在是 new_token
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)

                    #    - 新的右邻居：(new_token, word[i+1]) (注意：word[i+1] 是旧的 word[i+2])
                    if i < len(word)-1:
                        new_next = (new_token, word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                    # 合并后，索引 i 指向的是新 Token。
                    # i 不需要移动（i+=1），因为我们刚刚修改了 word[i] 并且删除了 word[i+1]。
                    # 下一轮循环会检查新的 (word[i], word[i+1])，即 (new_token, old_word[i+2])
                    # 这可以处理像 A A A -> X A 这样的情况，正确地更新新的邻居对

                else:
                    i += 1

        # 4d. 清理：移除已完全合并的 `best_pair`
        """
        防御性检查
        因为 best_pair 不一定还在 stats 和 indices 里。直接 del 可能抛 KeyError，所以要先判断。
        
        4c 循环中可能会触发删除pre_pair 或 next_pair的操作，
        如果 prev_pair 或 next_pair 恰好等于 best_pair，那么 stats[best_pair] 已经被删了。
        """
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    # --- 5. 构建最终的词表 ---
    # 添加 BPE 合并产生的 Token，ID 从 256 开始，按合并顺序递增
    for pair in merges:
        new_id = len(vocab)
        """
        注意：
        pair 本身是 tuple
        pair[0] + pair[1] 是 bytes 的拼接，结果是 bytes
        """
        vocab[new_id] = pair[0] + pair[1]

    # 添加特殊 Token
    for s_tok in special_tokens:
        new_id = len(vocab)
        s_bytes = s_tok.encode("utf-8")
        vocab[new_id] = s_bytes

    return vocab, merges


def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。

    为什么从 256 开始，而不是其他数字？
    256 即 U+0100。选择从 U+0100 开始是因为：
        U+0000 ~ U+00FF 已经被原始字节值占据（可打印的保留原码点，不可打印的我们不想用）；
        U+0100 之后的码点大多对应可打印的拉丁扩展字符，安全且连续；
        这样分配后，每个被“挪走”的字节都有一个确定的新码点，顺序与字节值顺序一致，便于实现和调试。
    """
    # 1. 初始收集“可直接保留”的字节
    # 合并后 bs 包含 188 个字节值，即：
    # [33, 34, ..., 126, 161, 162, ..., 172, 174, 175, ..., 255]
    # 这些字节对应的字符在 Unicode 中本来就是可打印的，所以希望它们保持原样映射
    """
    ord() 是 Python 的一个内置函数，用来把一个字符转换它对应的 Unicode 码点（整数）。
    基本用法：
    ord(c)
        c：必须是长度为 1 的字符串，也就是单个字符。
        返回值：该字符对应的 Unicode 码点整数。
    
    ord() 的反函数是 chr()
    
    一句话总结：
    ord() 是“字符 → 整数”，chr() 是“整数 → 字符”。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # cs 是 bs 的副本，目前存放的是与 bs 相同的整数，代表 Unicode 码点
    cs = bs[:]
    # 2. 为缺失的字节分配新码点
    """
    哪些字节不在初始 bs 中？一共 68 个：
        0 ~ 32：共 33 个（控制字符 + 空格）
        127 ~ 160：共 34 个（DEL + C1 控制字符）
        173：1 个（软连字符）
    这些缺失字节会按顺序被追加到 bs，并分配从 256 开始的新码点。
    """
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 3. 将码点转换为字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    """
    json.dump(obj, fp, ...)：
        将 Python 对象 json_vocab 序列化为 JSON 格式，并写入文件对象 f。
    json_vocab：
        是一个字典，形如 {0: "Ā", 1: "ā", 32: "Ġ", 256: "Ġthe", ...}，其中键是整数 token id，值是经过 bytes_to_unicode 映射后的可见字符串。
    indent=4：
        指定缩进为 4 个空格，使生成的 JSON 文件具有层次结构，便于人类阅读。如果不设置，JSON 会写成一整行，可读性差。
    """

    r"""
    需要注意的细节：
    1、整数键自动转为字符串
    JSON 标准要求键必须是字符串。json.dump 会自动把 json_vocab 中的整数键（如 0、256）转换为字符串键（"0"、"256"）。因此保存后的文件内容类似：
    json
    {
        "0": "Ā",
        "1": "ā",
        "32": "Ġ",
        "256": "Ġthe"
    }
    加载时需要用 int(k) 转回整数，这与之前加载代码中的 int(k) 正好对应。

    2、ensure_ascii 默认行为
    json.dump 默认 ensure_ascii=True，会把所有非 ASCII 字符转义为 \uXXXX 形式。例如 'Ġ' 会被写成 "\u0120"。虽然文件可读性降低，但 json.load 读取时会自动还原为 'Ġ'，不影响功能。如果想直接保存可见字符，可以加上 ensure_ascii=False：
    python
    json.dump(json_vocab, f, indent=4, ensure_ascii=False)
    3、文件覆盖
    由于使用了 "w" 模式，每次运行都会覆盖原有的 vocab.json，确保保存的是最新训练结果。
    """
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)

    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"  # 你的原始文本路径
    vocab_size = 10000  # 作业要求的词表大小
    # input_path = "data/owt_train.txt"
    # input_path = "data/chinese.txt"
    # vocab_size = 1000 # 作业要求的词表大小

    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")

    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)


if __name__ == "__main__":
    main()