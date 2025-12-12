import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import Config
from Dataset import TranslationDataset
from Transformer import ModernTransformer
from utils import calc_accuracy, MetricLogger, evaluate, get_cosine_schedule_with_warmup

def train():
    cfg = Config()
    print(f"Using device: {cfg.device}")
    
    # --- Data Loading ---
    print("Loading Training Data...")
    train_dataset = TranslationDataset(
        cfg.train_src, cfg.train_tgt, 
        cfg.vocab_src_path, cfg.vocab_tgt_path, 
        cfg.max_seq_len
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # Update config vocab_size based on actual loaded vocab
    # We take the max to ensure the embedding layer covers both source and target indices
    cfg.vocab_size = max(len(train_dataset.src_tokenizer), len(train_dataset.tgt_tokenizer))
    print(f"Vocab size updated to: {cfg.vocab_size}")

    print("Loading Test Data...")
    # Ensure test files exist, otherwise warn
    try:
        test_dataset = TranslationDataset(
            cfg.test_src, cfg.test_tgt, 
            cfg.vocab_src_path, cfg.vocab_tgt_path, 
            cfg.max_seq_len
        )
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    except FileNotFoundError:
        print("Warning: Test files not found. Validation will be skipped.")
        test_dataloader = None

    # --- Model Init ---
    model = ModernTransformer(cfg).to(cfg.device)
    
    try:
        model.load_pretrained_embeddings(
            cfg.w2v_src_path, cfg.w2v_tgt_path,
            train_dataset.src_tokenizer, train_dataset.tgt_tokenizer
        )
        model.freeze_embeddings()
        print("Pretrained embeddings loaded and frozen.")
    except Exception as e:
        print(f"Warning: Could not load Word2Vec embeddings ({e}). Training from scratch.")

    # --- Setup ---
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    total_steps = len(train_dataloader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tgt_pad_id, label_smoothing=cfg.label_smoothing)
    
    # Initialize Logger
    best_acc = 0.0
    logger = MetricLogger(cfg.log_file)
    
    # --- Training Loop ---
    for epoch in range(cfg.epochs):
        if epoch == int(cfg.epochs * 0.3):
            model.unfreeze_embeddings()
            print("Unfroze embeddings for fine-tuning.")
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_tokens = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for batch in progress_bar:
            src = batch["src"].to(cfg.device)
            tgt = batch["tgt"].to(cfg.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = model.create_masks(src, tgt_input, train_dataset.src_pad_id, train_dataset.tgt_pad_id)
            
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            flat_logits = logits.reshape(-1, cfg.vocab_size)
            flat_targets = tgt_output.reshape(-1)
            
            loss = criterion(flat_logits, flat_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track Metrics
            epoch_loss += loss.item()
            n_correct, n_total = calc_accuracy(flat_logits, flat_targets, train_dataset.tgt_pad_id)
            epoch_correct += n_correct
            epoch_tokens += n_total
            
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # End of Epoch Aggregation
        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_acc = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0
        
        # Validation
        val_loss, val_acc, val_bleu = 0, 0, 0
        if test_dataloader:
            # Pass the full vocab object (tgt_tokenizer) to evaluate for BLEU decoding
            val_loss, val_acc, val_bleu = evaluate(
                model, test_dataloader, criterion, 
                train_dataset.tgt_tokenizer, cfg.device
            )
        
        # Log to CSV and Console
        logger.log(epoch+1, avg_train_loss, avg_train_acc, val_loss, val_acc, val_bleu)
        
        # Save best model (using BLEU is often better for MT, but Acc is stable)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best validation accuracy: {best_acc*100:.2f}% (BLEU: {val_bleu:.2f}) - saving model.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'bleu': val_bleu
            }, cfg.model_save_path)

    print(f"Training complete. Logs saved to {cfg.log_file}")

if __name__ == "__main__":
    train()