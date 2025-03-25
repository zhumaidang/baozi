"""
    Byte-Pair Encoding (BPE)
    核心步骤：
    1. 计算初始词表：通过训练语料获得或者最初的英文中26个字母加上各种符号以及常见中文字符，这些作为初始词表
    2. 构建频率统计：统计所有子词单元对（两个连续的子词）在文本中出现频率
    3. 合并频率最高的子词对：选择出现频率最高的子词对，将它们合并成一个新的子词单元，并更新词汇表
    4. 重复合并步骤：不断重复步骤2和步骤3，直到满足预设的词汇表大小、合并次数或者达到停止条件（不再有意义的合并）
    5. 分词：使用最终得到的词汇表对文本进行分词
"""

from collections import defaultdict

class BPE:
    def __init__(self):
        self.stats = defaultdict(int)
        self.splits = {}


    # 构建频率统计
    def build_stats(self, sentences):
        for sentence in sentences:
            symbols = sentence.split()
            for symbol in symbols:
                self.stats[symbol] += 1

        alphabet = []
        for word in self.stats.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        vocab = alphabet.copy()  # 初始词表

        return vocab


    # 根据初始词表拆分每个词
    def build_splites(self):
        splits = {word: [c for c in word] for word in self.stats.keys()}
        
        return splits

    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in self.stats.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    # 合并频率最高的子词对
    def merge_pair(self, a, b, splits):
        for word in self.stats:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def main(self, sentences):
        merges = {}
        vocab_size = 50
        vocab = self.build_stats(sentences)  # 初始词表
        splits = self.build_splites()  # 拆分后的词表

        while len(vocab) < vocab_size:
            pair_freqs = self.compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = self.merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

        return vocab


if __name__ == "__main__":
    bpe = BPE()
    # 语料
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
    vocab = bpe.main(sentences)
    print(vocab)

