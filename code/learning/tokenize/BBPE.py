"""
    Byte-level BPE (BBPE)
    核心步骤：
    1. 计算初始词表：构建初始词表，包含一个字节的所有表示（256）
    2. 构建频率统计：统计所有子词单元对（两个连续的子词）在文本中的出现频率
    3. 合并频率最高的子词对：选择出现频率最高的子词对，将他们合并成一个新的子词单元，并更新词汇表
    4. 重复合并步骤：不断重复步骤2和步骤3，直到满足预设的词汇表大小、合并次数或者达到停止条件（不再有意义的合并）
    5. 分词：使用最终得到的词汇表对文本进行分词

    优点：
    1. 跨语言通用性
    2. 减少词汇表大小
    3. 处理罕见字符OOV：因为它不会为每个罕见字符分配单独的词汇表条目，而是将它们作为字节序列处理
"""

from collections import defaultdict
sentences = [
    "我",
    "喜欢",
    "吃",
    "苹果",
    "他",
    "不",
    "喜欢",
    "吃",
    "苹果派",
    "I like to eat apples",
    "She has a cute cat",
    "you are very cute",
    "give you a hug",
]
# 构建初始词汇表，包含一个字节的256个表示
initial_vocab = [bytes([byte]) for byte in range(256)]
vocab = initial_vocab.copy()
print("initial_vocab:", initial_vocab)

# 构建频率统计
def build_stats(sentences):
    stats = defaultdict(int)
    for sentence in sentences:
        symbols = sentence.split()
        for symbol in symbols:
            stats[symbol.encode("utf-8")] += 1
    return stats
stats = build_stats(sentences)

print(stats)
splits = {word: [byte for byte in word] for word in stats.keys()}
print("--------------------------")
print("splits:", splits)

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in stats.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

def merge_pair(pair, splits):
    merged_byte = bytes(pair)
    for word in stats:
        split = splits[word]
        print("++++++>>>", split)
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            print('--------------------------===>>>', pair)
            if split[i:i+2] == pair:  # 检查分割中是否有这对字节
                split = split[:i] + [merged_byte] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

vocab_size = 50
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ()
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(best_pair, splits)
    merged_byte = bytes(best_pair)

print("vocab:", vocab)