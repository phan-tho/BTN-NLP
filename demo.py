import torch
from Config import Config
from Transformer import ModernTransformer
from Vocab import SimpleVocab
from utils import beam_search_decode
import argparse

def translate(text, model, src_vocab, tgt_vocab, cfg):
    model.eval()
    
    newl = ''
    for c in text:
        if c.isalnum() or c.isspace():
            newl += c
        elif c in ".,?":
            newl += ' ' + c + ' '

    src_ids = src_vocab.encode(newl)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(cfg.device) # (1, seq_len)
    
    # (1, 1, 1, seq_len)
    src_mask = (src_tensor != src_vocab.pad_id).unsqueeze(1).unsqueeze(2)
    
    # 4. Inference
    with torch.no_grad():
        output_ids = beam_search_decode(
            model, 
            src_tensor, 
            src_mask, 
            max_len=cfg.max_seq_len, 
            start_symbol=tgt_vocab.sos_id, 
            end_symbol=tgt_vocab.eos_id, 
            beam_size=cfg.beam_size, 
            device=cfg.device
        )
    
    # 5. Decode
    output_words = tgt_vocab.decode(output_ids)
    
    filtered_words = []
    for w in output_words:
        if w not in ["[SOS]", "[EOS]", "[PAD]"]:
            filtered_words.append(w)
            
    return " ".join(filtered_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a trained Transformer model.")
    parser.add_argument("text", type=str, default="Hello world. I am fine, thank you. And you?")
    args = parser.parse_args()

    cfg = Config()
    
    print("Loading Vocabularies...")
    src_vocab = SimpleVocab()
    src_vocab.load(cfg.vocab_src_path)
    tgt_vocab = SimpleVocab()
    tgt_vocab.load(cfg.vocab_tgt_path)
    
    model = ModernTransformer(cfg).to(cfg.device)
        
    try:
        print(f"Loading model weights from {cfg.model_save_path}...")
        checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model checkpoint not found. Please train the model first.")
        exit()
    
    print("Translating...")
    translation = translate(args.text, model, src_vocab, tgt_vocab, cfg)
    
    print(f"Original:   {args.text}")
    print(f"Translated: {translation}")
