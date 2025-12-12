import os
from gensim.models import Word2Vec
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

def train_word2vec(txt_file, tokenizer, model_path, vector_size):
    print(f"Training Word2Vec for {txt_file}...")
    
    # Generator to yield tokenized sentences line by line
    class SentenceIterator:
        def __iter__(self):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        # Simple split for Word2Vec to match SimpleVocab
                        tokens = line.strip().split()
                        yield tokens

    sentences = SentenceIterator()
    
    # Train Word2Vec
    # window=5, min_count=1, workers=4
    w2v_model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        window=5, 
        min_count=1, 
        workers=4,
        sg=1 # Skip-gram is usually better for smaller datasets
    )
    
    w2v_model.save(model_path)
    print(f"Saved Word2Vec model to {model_path}")

if __name__ == "__main__":
    cfg = Config()
    
    # 1. Train Tokenizers (Vocabs)
    if not os.path.exists(cfg.train_src):
        print(f"Error: Data file {cfg.train_src} not found.")
        exit()

    # Note: vocab_size in Config is ignored here, size depends on min_freq
    tok_en = build_vocab([cfg.train_src], cfg.vocab_src_path, min_freq=5)
    tok_vi = build_vocab([cfg.train_tgt], cfg.vocab_tgt_path, min_freq=5)
    
    # 2. Train Word2Vec on the tokens
    # We use d_model size so we can load it directly into the Transformer
    train_word2vec(cfg.train_src, tok_en, cfg.w2v_src_path, cfg.d_model)
    train_word2vec(cfg.train_tgt, tok_vi, cfg.w2v_tgt_path, cfg.d_model)
    
    print("Preprocessing complete.")