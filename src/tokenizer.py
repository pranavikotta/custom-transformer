'''
tokenizer.py: Custom Byte-Pair Encoding (BPE) tokenizer implementation.
This text tokenization technique chunks words into subwords, aiming to 
find common pairs of characters until the desired vocabulary size is 
reached. This process simplifies and optimizes the model's ability to
understand a wide variety of words. The encode and decoding methods will
handle new words while the other methods are meant for training.
'''
from collections import defaultdict

class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = defaultdict(int)
        self.merges = []
        self.int_to_string = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    def _get_freqs(self, word_freqs):
        #receives a dictionary of words represented as tuples of characters and their counts
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += freq
        return pair_counts
    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < (len(word)-1) and (word[i], word[i+1]) == pair:
                    new_word.append(pair[0] + pair[1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_word_freqs[tuple(new_word)] += freq
        return new_word_freqs
    def clean_text(self, text):
        cleaned_text = text.replace("\n", " ")
        cleaned_text = cleaned_text.replace("\t", " ")
        words = cleaned_text.split(" ")
        return words
    def train(self, text):
        #model learns the BPE merges based on the frequency of character pairs in the input text
        words = self.clean_text(text)
        #track frequency of each word as a tuple of characters, prepending a special character to denote the start of a word
        word_freqs = defaultdict(int)
        for word in words:
            if word:
                word_freqs[tuple(f"Ġ{word}")] += 1
        #ensure all unique characters are stored in the vocab before adding merged pairs
        unique_chars = set()
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            unique_chars.add(token)
        for word in word_freqs.keys():
            for char in word:
                unique_chars.add(char)
        self.vocab = {char: i for i, char in enumerate(sorted(list(unique_chars)))}
        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        self.bos_id = self.vocab[self.bos_token]
        self.eos_id = self.vocab[self.eos_token]
        
        #iteratively merge the most common pairs until the vocab size is reached
        while len(self.vocab) < self.vocab_size:
            pair_counts = self._get_freqs(word_freqs)
            if not pair_counts:
                break
            common_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(common_pair)
            word_freqs = self._merge_pair(common_pair, word_freqs)
            self.vocab[common_pair[0] + common_pair[1]] = len(self.vocab)
        self.int_to_string = {i: token for token, i in self.vocab.items()}
    def decode(self, token_ids):
        tokens = [self.int_to_string[token_id] for token_id in token_ids]
        string = "".join(tokens).replace("Ġ", " ")
        return string
    def encode(self, text):
        words = self.clean_text(text)
        char_list = []
        for word in words:
            if word:
                char_list.append(tuple(f"Ġ{word}"))
        for pair in self.merges:
            for i in range(len(char_list)):
                j = 0
                while (j < len(char_list[i]) -1):
                    if (char_list[i][j], char_list[i][j+1]) == pair:
                        char_list[i] = char_list[i][:j] + (pair[0] + pair[1],) + char_list[i][j+2:]
                    else:
                        j+=1
        token_ids = []
        for word_tuple in char_list:
            for subword in word_tuple:
                token_ids.append(self.vocab.get(subword, self.unk_id))
        return token_ids

#test tokenizer
if __name__ == "__main__":
    text = "hello world! this is a test of the BPE tokenizer, to see if it can handle new words like tokenizer and tokenization."
    bpe = BPE(vocab_size=50)
    bpe.train(text)
    test_text = "hello tokenizer, this is the number 13."
    encoded = bpe.encode(test_text)
    print("Encoded:", encoded)
    decoded = bpe.decode(encoded)
    print("Decoded:", decoded)