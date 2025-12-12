import torch

class Config:
    # root = 
    
    # Paths
    train_src = "/kaggle/input/iwslt15-englishvietnamese/IWSLT'15 en-vi/train.en.txt" # Replace with your actual path
    train_tgt = "/kaggle/input/iwslt15-englishvietnamese/IWSLT'15 en-vi/train.vi.txt"  # Replace with your actual path

    test_src = "/kaggle/input/iwslt15-englishvietnamese/IWSLT'15 en-vi/tst2013.en.txt"
    test_tgt = "/kaggle/input/iwslt15-englishvietnamese/IWSLT'15 en-vi/tst2013.vi.txt"
    
    log_file = 'training_log.csv'
    
    # Tokenizer & Vocab
    vocab_src_path = 'vocab_en.json'
    vocab_tgt_path = 'vocab_vi.json'
    w2v_src_path = 'w2v_en.model'
    w2v_tgt_path = 'w2v_vi.model'
    vocab_size = 16000  # Smaller vocab for IWSLT is usually sufficient
    
    # Model Architecture (Modern "Small" Config for IWSLT/Single GPU)
    d_model = 512
    n_layers = 6       # 6 Encoder, 6 Decoder
    n_heads = 8
    d_ff = int(2 * 4 * d_model / 3) # SwiGLU hidden size (approx 2/3 of 4d)
    dropout = 0.1
    max_seq_len = 128  # IWSLT sentences are short
    
    # Training
    batch_size = 64
    lr = 5e-4
    weight_decay = 0.01
    epochs = 20
    warmup_steps = 4000
    label_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inference
    beam_size = 4
    
    # Checkpointing
    model_save_path = 'transformer_en_vi.pth'