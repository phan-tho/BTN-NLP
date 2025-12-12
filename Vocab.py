import json
from collections import Counter

class SimpleVocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.specials = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        
    def build(self, filepath, min_freq=5):
        print(f"Building vocab from {filepath} with min_freq={min_freq}...")
        counter = Counter()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                newl = ''
                for c in line:
                    if c.isalnum() or c.isspace():
                        newl += c
                    elif c in ".,?":
                        newl += ' ' + c + ' '

                words = newl.strip().split()
                counter.update(words)
        
        # Start with specials
        self.word2idx = {k: i for i, k in enumerate(self.specials)}
        idx = len(self.specials)
        
        for word, count in counter.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"Vocab built. Size: {len(self.word2idx)}")

    def encode(self, text):
        words = text.strip().split()
        return [self.word2idx.get(w, self.word2idx["[UNK]"]) for w in words]

    def decode(self, ids):
        if isinstance(ids, list):
            return [self.idx2word.get(i, "[UNK]") for i in ids]
        return self.idx2word.get(ids, "[UNK]")

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"Vocab saved to {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v) if isinstance(v, str) and v.isdigit() else v: k for k, v in self.word2idx.items()} # Handle json keys if saved inversely, but here word2idx has string keys
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def __len__(self):
        return len(self.word2idx)
        
    def pad_id(self): return self.word2idx["[PAD]"]
    def unk_id(self): return self.word2idx["[UNK]"]
    def sos_id(self): return self.word2idx["[SOS]"]
    def eos_id(self): return self.word2idx["[EOS]"]
