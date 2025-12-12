import torch
from torch.utils.data import Dataset
from Vocab import SimpleVocab
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab_path, tgt_vocab_path, max_len):
        self.src_data = [line.strip() for line in open(src_file, 'r', encoding='utf-8').readlines() if line.strip()]
        self.tgt_data = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8').readlines() if line.strip()]
        # 133168
        self.src_data = self.src_data[:130000]
        self.tgt_data = self.tgt_data[:130000]
        
        assert len(self.src_data) == len(self.tgt_data), "Source and Target files must have same number of lines"
        
        self.src_tokenizer = SimpleVocab()
        self.src_tokenizer.load(src_vocab_path)
        
        self.tgt_tokenizer = SimpleVocab()
        self.tgt_tokenizer.load(tgt_vocab_path)
        
        # Access dictionary directly to ensure we get integers, not methods
        self.src_pad_id = self.src_tokenizer.word2idx.get("[PAD]", 0)
        self.tgt_pad_id = self.tgt_tokenizer.word2idx.get("[PAD]", 0)
        self.tgt_sos_id = self.tgt_tokenizer.word2idx.get("[SOS]", 1)
        self.tgt_eos_id = self.tgt_tokenizer.word2idx.get("[EOS]", 2)
        
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # Encode
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_enc_ids = self.tgt_tokenizer.encode(tgt_text)
        
        # Add SOS/EOS to target
        tgt_ids = [self.tgt_sos_id] + tgt_enc_ids + [self.tgt_eos_id]
        
        # Truncate
        if len(src_ids) > self.max_len: src_ids = src_ids[:self.max_len]
        if len(tgt_ids) > self.max_len: tgt_ids = tgt_ids[:self.max_len]
        
        # Padding
        src_mask = [1] * len(src_ids) + [0] * (self.max_len - len(src_ids))
        src_ids = src_ids + [self.src_pad_id] * (self.max_len - len(src_ids))
        
        tgt_padding_len = self.max_len - len(tgt_ids)
        tgt_ids = tgt_ids + [self.tgt_pad_id] * tgt_padding_len
        
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long).unsqueeze(0).unsqueeze(0), # (1, 1, seq_len) for broadcasting
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            # For target mask we need causal masking later in model, here just padding mask
            "tgt_pad_mask": (torch.tensor(tgt_ids, dtype=torch.long) != self.tgt_pad_id).unsqueeze(0).unsqueeze(0)
        }