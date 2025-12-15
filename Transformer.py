import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. RMSNorm---
# Impact: RMSNorm simplifies LayerNorm by removing mean centering. 
# It is computationally cheaper and often leads to better training stability and convergence.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        return x + self.pe[:, :x.size(1)]

# --- 3. SwiGLU Feed Forward ---
# Impact: Replaces the standard ReLU/GELU Feed Forward Network.
# SwiGLU (Swish-Gated Linear Unit) has been shown to offer better performance 
# and learning capacity than standard FFNs in LLMs (e.g., LLaMA, PaLM).
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # F.silu is Swish
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# --- 4. Attention (Standard Scaled Dot-Product) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_x=None):
        # kv_x is for cross-attention (encoder output)
        B, Seq, _ = x.shape
        
        xq = self.wq(x).view(B, Seq, self.n_heads, self.head_dim)
        
        if kv_x is None: 
            # Self Attention
            xk = self.wk(x).view(B, Seq, self.n_heads, self.head_dim)
            xv = self.wv(x).view(B, Seq, self.n_heads, self.head_dim)
        else:
            # Cross Attention
            B_kv, Seq_kv, _ = kv_x.shape
            xk = self.wk(kv_x).view(B_kv, Seq_kv, self.n_heads, self.head_dim)
            xv = self.wv(kv_x).view(B_kv, Seq_kv, self.n_heads, self.head_dim)

        # Transpose for dot prod: (B, n_heads, Seq, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, xv)
        output = output.transpose(1, 2).contiguous().view(B, Seq, -1)
        
        return self.wo(output)

# --- 5. Encoder & Decoder Layers ---
class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, mask):
        # Pre-Norm
        h = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = h + x # Residual
        
        h = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = h + x
        return x

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        
        self.norm2 = RMSNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        
        self.norm3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        # 1. Self Attention
        h = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=tgt_mask)
        x = h + x
        
        # 2. Cross Attention
        h = x
        x = self.norm2(x)
        x = self.cross_attn(x, mask=src_mask, kv_x=enc_out)
        x = h + x
        
        # 3. FFN
        h = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = h + x
        return x

# --- 6. Main Model ---
class ModernTransformer(nn.Module):
    def __init__(self, cfg, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.src_embed = nn.Embedding(self.src_vocab_size, cfg.d_model)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, cfg.d_model)
        
        # Standard Positional Encoding
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        
        # Layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        
        self.final_norm = RMSNorm(cfg.d_model)
        self.fc_out = nn.Linear(cfg.d_model, self.tgt_vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def load_pretrained_embeddings(self, src_w2v_path, tgt_w2v_path, src_vocab, tgt_vocab):
        """
        Loads PyTorch Skip-gram weights into the Embedding layers.
        """
        print("Loading Pretrained Embeddings...")
        
        def load_emb(path, embedding_layer):
            try:
                ckt = torch.load(path, map_location='cpu')
                weights = ckt['weight']

                # with torch.no_grad():
                #     rows = min(weights.shape[0], embedding_layer.weight.shape[0])
                #     cols = min(weights.shape[1], embedding_layer.weight.shape[1])
                #     embedding_layer.weight[:rows, :cols] = weights[:rows, :cols]
                with torch.no_grad():
                    embedding_layer.weight = weights
                    
                print(f"Loaded embeddings from {path}")
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        load_emb(src_w2v_path, self.src_embed)
        load_emb(tgt_w2v_path, self.tgt_embed)

    def freeze_embeddings(self):
        self.src_embed.weight.requires_grad = False
        self.tgt_embed.weight.requires_grad = False
    def unfreeze_embeddings(self):
        self.src_embed.weight.requires_grad = True
        self.tgt_embed.weight.requires_grad = True

    def encode(self, src, src_mask):
        # Apply embedding + pos encoding
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        # Apply embedding + pos encoding
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
            
        logits = self.fc_out(self.final_norm(x))
        return logits

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)

    def create_masks(self, src, tgt, pad_idx_src, pad_idx_tgt):
        # src shape: (B, Src_Seq)
        # tgt shape: (B, Tgt_Seq)

        # Source mask (PAD mask)
        
        src_mask = (src != pad_idx_src).unsqueeze(1).unsqueeze(2) # (B, 1, 1, Seq)
        
        # Target mask (Causal + PAD)
        tgt_pad_mask = (tgt != pad_idx_tgt).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        
        tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0)
        
        return src_mask, tgt_mask