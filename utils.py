import torch
import csv
import os
import math

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
        self.headers = ["Epoch", "Train_Loss", "Train_Acc", "Train_PPL", "Val_Loss", "Val_Acc", "Val_PPL"]
        
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
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
            f"{val_ppl:.4f}"
        ]
        
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"\n[Report] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val PPL: {val_ppl:.2f}")