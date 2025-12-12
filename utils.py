import torch
import csv
import os
import math
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def batch_greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """
    Batched greedy decoding for fast validation.
    """
    bs = src.size(0)
    tgt = torch.full((bs, 1), start_symbol, dtype=torch.long, device=device)
    
    # Encode once
    enc_out = model.encode(src, src_mask)
    
    finished = torch.zeros(bs, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        # Create mask for current tgt
        _, tgt_mask = model.create_masks(src, tgt, -1, -1) # Pad IDs don't matter for mask generation logic here usually, but better safe
        # Actually create_masks needs pad ids. We can just pass 0 or handle it. 
        # Let's rely on the fact that we are generating, so no padding in tgt yet except what we might add?
        # Simplified: Just causal mask is needed for decoding usually, but our create_masks does both.
        # Let's manually create causal mask to be safe and fast
        sz = tgt.size(1)
        causal_mask = torch.tril(torch.ones((sz, sz), device=device)).bool().unsqueeze(0).unsqueeze(0)
        
        out = model.decode(tgt, enc_out, src_mask, causal_mask)
        logits = out[:, -1, :]
        next_word = torch.argmax(logits, dim=-1, keepdim=True)
        
        tgt = torch.cat([tgt, next_word], dim=1)
        
        # Check for EOS
        is_eos = (next_word.squeeze(-1) == end_symbol)
        finished = finished | is_eos
        
        if finished.all():
            break
            
    return tgt

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, beam_size, device):
    """
    Beam Search for a single sentence (Inference).
    """
    # Ensure src is (1, seq_len)
    enc_out = model.encode(src, src_mask)
    
    # (score, sequence)
    beams = [(0.0, [start_symbol])]
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == end_symbol:
                candidates.append((score, seq))
                continue
                
            tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            sz = tgt_tensor.size(1)
            causal_mask = torch.tril(torch.ones((sz, sz), device=device)).bool().unsqueeze(0).unsqueeze(0)
            
            # We only need the last token prediction
            out = model.decode(tgt_tensor, enc_out, src_mask, causal_mask)
            logits = out[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            topk_probs, topk_ids = torch.topk(log_probs, beam_size)
            
            for i in range(beam_size):
                sym = topk_ids[0, i].item()
                val = topk_probs[0, i].item()
                candidates.append((score + val, seq + [sym]))
        
        # Sort and keep top k
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # Early stop if all top beams end with EOS
        if all(b[1][-1] == end_symbol for b in beams):
            break
            
    return beams[0][1]

def compute_bleu(preds, targets, vocab):
    """
    Computes Corpus BLEU-4 score manually without external libraries.
    preds: list of list of token IDs
    targets: list of list of token IDs
    vocab: SimpleVocab object
    """
    # 1. Decode and strip specials
    pred_tokens = []
    ref_tokens = []
    
    specials = {vocab.pad_id, vocab.sos_id, vocab.eos_id}
    
    for p in preds:
        words = [vocab.idx2word[i] for i in p if i not in specials]
        pred_tokens.append(words)
        
    for t in targets:
        words = [vocab.idx2word[i] for i in t if i not in specials]
        ref_tokens.append(words)

    # 2. Calculate N-gram precisions (N=1 to 4)
    precisions = []
    for n in range(1, 5):
        counts = 0
        total = 0
        for cand, ref in zip(pred_tokens, ref_tokens):
            cand_ngrams = Counter([tuple(cand[i:i+n]) for i in range(len(cand)-n+1)])
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
            
            for gram, count in cand_ngrams.items():
                counts += min(count, ref_ngrams.get(gram, 0))
                total += sum(cand_ngrams.values()) # Total n-grams in candidate
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(counts / total)

    # 3. Geometric Mean
    if min(precisions) > 0:
        p_log_sum = sum((1/4) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    # 4. Brevity Penalty
    c = sum(len(cand) for cand in pred_tokens)
    r = sum(len(ref) for ref in ref_tokens)
    
    if c == 0: return 0.0
    
    bp = math.exp(1 - r/c) if c < r else 1.0
    
    return bp * geo_mean * 100 # Return as percentage

def evaluate(model, dataloader, criterion, vocab, device):
    """
    Runs validation. Returns avg_loss, avg_acc, and bleu_score.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    all_preds = []
    all_targets = []
    
    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    vocab_size = len(vocab)
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = model.create_masks(src, tgt_input, pad_id, pad_id)
            
            # 1. Teacher Forcing for Loss & Accuracy
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = tgt_output.reshape(-1)
            
            loss = criterion(flat_logits, flat_targets)
            total_loss += loss.item()
            
            n_correct, n_total = calc_accuracy(flat_logits, flat_targets, pad_id)
            total_correct += n_correct
            total_tokens += n_total
            
            # 2. Greedy Decoding for BLEU (Faster than Beam for validation)
            # We use the same src and src_mask
            # Max len can be heuristic, e.g., src_len + 50
            max_len = src.size(1) + 50
            generated_ids = batch_greedy_decode(model, src, src_mask, max_len, sos_id, eos_id, device)
            
            all_preds.extend(generated_ids.tolist())
            all_targets.extend(tgt.tolist()) # Use full tgt for reference
            
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
    
    # Calculate BLEU
    bleu_score = compute_bleu(all_preds, all_targets, vocab)
    
    return avg_loss, avg_acc, bleu_score

def calc_accuracy(logits, targets, pad_id):
    """
    Calculates token-level accuracy, ignoring padding.
    """
    # logits: (batch_size * seq_len, vocab_size)
    # targets: (batch_size * seq_len)
    
    preds = torch.argmax(logits, dim=-1)
    
    # Create mask for non-padding tokens
    mask = targets != pad_id
    
    # Calculate correct predictions only on non-padding tokens
    correct = (preds == targets) & mask
    
    num_correct = correct.sum().item()
    num_total = mask.sum().item()
    
    return num_correct, num_total

class MetricLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.headers = ["Epoch", "Train_Loss", "Train_Acc", "Train_PPL", "Val_Loss", "Val_Acc", "Val_PPL", "Val_BLEU"]
        
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, val_bleu):
        # Calculate Perplexity (PPL) safely
        train_ppl = math.exp(min(train_loss, 100)) # Cap to prevent overflow
        val_ppl = math.exp(min(val_loss, 100))
        
        row = [
            epoch, 
            f"{train_loss:.4f}", 
            f"{train_acc:.4f}", 
            f"{train_ppl:.4f}",
            f"{val_loss:.4f}", 
            f"{val_acc:.4f}", 
            f"{val_ppl:.4f}",
            f"{val_bleu:.2f}"
        ]
        
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"\n[Evaluate] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val BLEU: {val_bleu:.2f}")