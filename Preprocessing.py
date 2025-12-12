import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from gensim.models import Word2Vec # Removed
from Config import Config
from Vocab import SimpleVocab
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def build_vocab(files, vocab_path, min_freq=5):
    # files is a list, we take the first one for simplicity as per SimpleVocab design
    vocab = SimpleVocab()
    vocab.build(files[0], min_freq=min_freq)
    vocab.save(vocab_path)
    return vocab

class SkipGramDataset(Dataset):
    def __init__(self, txt_file, vocab, window_size=2):
        self.pairs = []
        print(f"Generating Skip-gram pairs for {txt_file}...")
        # Assuming vocab has word2idx
        word2idx = vocab.word2idx
        unk_idx = word2idx.get('<UNK>', 0)
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                tokens = line.strip().split()
                token_ids = [word2idx.get(t, unk_idx) for t in tokens]
                
                for i, center in enumerate(token_ids):
                    start = max(0, i - window_size)
                    end = min(len(token_ids), i + window_size + 1)
                    for j in range(start, end):
                        if i != j:
                            self.pairs.append((center, token_ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        initrange = 0.5 / embed_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, center, context, negatives):
        u = self.in_embed(center) # (B, D)
        v = self.out_embed(context) # (B, D)
        n = self.out_embed(negatives) # (B, K, D)
        
        pos_score = torch.sum(u * v, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
        
        neg_score = torch.bmm(n, u.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).sum(1)
        
        return (pos_loss + neg_loss).mean()

def train_skipgram(txt_file, vocab, model_path, vector_size, epochs=3, batch_size=1024, window=2, neg_samples=5):
    print(f"Training Skip-gram for {txt_file}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SkipGramDataset(txt_file, vocab, window)
    if len(dataset) == 0:
        print("No data found.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SkipGramModel(len(vocab.word2idx), vector_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center = center.to(device)
            context = context.to(device)
            negatives = torch.randint(0, len(vocab.word2idx), (center.size(0), neg_samples)).to(device)
            
            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(dataloader):.4f}")
    
    # Save state dict (only input embeddings usually needed)
    torch.save({'weight': model.in_embed.weight.data}, model_path)
    print(f"Saved Skip-gram embeddings to {model_path}")

if __name__ == "__main__":
    cfg = Config()
    
    # 1. Train Tokenizers (Vocabs)
    if not os.path.exists(cfg.train_src):
        print(f"Error: Data file {cfg.train_src} not found.")
        exit()

    # Note: vocab_size in Config is ignored here, size depends on min_freq
    tok_en = build_vocab([cfg.train_src], cfg.vocab_src_path, min_freq=5)
    tok_vi = build_vocab([cfg.train_tgt], cfg.vocab_tgt_path, min_freq=5)
    
    # 2. Train Skip-gram on the tokens
    train_skipgram(cfg.train_src, tok_en, cfg.w2v_src_path, cfg.d_model)
    train_skipgram(cfg.train_tgt, tok_vi, cfg.w2v_tgt_path, cfg.d_model)
    
    print("Preprocessing complete.")