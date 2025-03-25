"""
    Wordpiece核心步骤：
    1. 计算初始此表：通过训练语料获得或者最初的英文中26个字母加上各种符号以及常见中文字符，这些作为初始此表
    2. 计算合并分数：对训练语料拆分的子词单元通过计算互信息分数，找到分数最高的子词对
    3. 合并分数最高的子词对：选择分数最高的子词对，合并为一个新的子词单元，并更新词表
    4. 重复合并步骤：不断重复步骤2和步骤3，直到达到指定的词表大小或不再有可合并的子词对
    5. 分词：使用最终得到的词表对文本进行分词

"""

from collections import defaultdict

class word_piece:
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
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        alphabet.sort()
        # 初始词表
        vocab = alphabet.copy()
        return vocab

    # 根据初始词表拆分每个词
    def build_splites(self):
        self.splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.stats.keys()
        }
        return self.splits

    # 计算互信息分数
    def compute_pair_scores(self, splits):
        letter_freqs = defaultdict(int)  # 单个元素的频数
        pair_freqs = defaultdict(int)  # 合并对的频数
        for word, freq in self.stats.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    # 查找分数最高的子词对
    def find_best_pair(self):
        best_pair = ""
        max_score = None
        pair_scores = self.compute_pair_scores(self.splits)
        for pair, score in pair_scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score

    # 合并分数最高的子词对
    def merge_pair(self, a, b, splits):
        for word in self.stats:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    # 循环迭代，直至vocabulary达到指定大小
    def main(self, sentences):
        vocab_size = 50
        vocab = self.build_stats(sentences)
        splits = self.build_splites()
        while len(vocab) < vocab_size:
            scores = self.compute_pair_scores(splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits = self.merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            vocab.append(new_token)
        return vocab
    
if __name__ == "__main__":
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
    word_piece = word_piece()
    print(word_piece.main(sentences))
