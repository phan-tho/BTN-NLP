import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gensim.models import Word2Vec

# --- 1. RMSNorm (Pre-normalization preferred) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

# --- 2. Rotary Positional Embeddings (RoPE) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq shape: (batch, seq_len, n_heads, head_dim) -> reshape to pairs for complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs for broadcasting (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, xq_.size(-1))
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- 3. SwiGLU Feed Forward ---
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

# --- 4. Attention (with RoPE) ---
class ModernAttention(nn.Module):
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

    def forward(self, x, freqs_cis, mask=None, kv_x=None):
        # kv_x is for cross-attention (encoder output)
        B, Seq, _ = x.shape
        
        xq = self.wq(x).view(B, Seq, self.n_heads, self.head_dim)
        
        if kv_x is None: 
            # Self Attention
            xk = self.wk(x).view(B, Seq, self.n_heads, self.head_dim)
            xv = self.wv(x).view(B, Seq, self.n_heads, self.head_dim)
            # Apply RoPE only for self-attention queries and keys
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        else:
            # Cross Attention (No RoPE usually on Cross Attn, or only on Query)
            # Standard practice: No positional encoding on Cross Attn Keys/Values derived from Encoder
            B_kv, Seq_kv, _ = kv_x.shape
            xk = self.wk(kv_x).view(B_kv, Seq_kv, self.n_heads, self.head_dim)
            xv = self.wv(kv_x).view(B_kv, Seq_kv, self.n_heads, self.head_dim)
            # We don't apply RoPE to cross attention in standard transformer recipes usually,
            # as position info is already in the encoder states.

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
        self.attn = ModernAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, freqs_cis, mask):
        # Pre-Norm
        h = x
        x = self.norm1(x)
        x = self.attn(x, freqs_cis, mask)
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
        self.self_attn = ModernAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        
        self.norm2 = RMSNorm(cfg.d_model)
        self.cross_attn = ModernAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        
        self.norm3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, enc_out, freqs_cis, tgt_mask, src_mask):
        # 1. Self Attention
        h = x
        x = self.norm1(x)
        x = self.self_attn(x, freqs_cis, tgt_mask)
        x = h + x
        
        # 2. Cross Attention
        h = x
        x = self.norm2(x)
        x = self.cross_attn(x, freqs_cis=None, mask=src_mask, kv_x=enc_out)
        x = h + x
        
        # 3. FFN
        h = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = h + x
        return x

# --- 6. Main Model ---
class ModernTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        
        # Embeddings
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(cfg.d_model // cfg.n_heads, cfg.max_seq_len * 2)
        
        # Layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        
        self.final_norm = RMSNorm(cfg.d_model)
        self.fc_out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def load_pretrained_embeddings(self, src_w2v_path, tgt_w2v_path, src_vocab, tgt_vocab):
        """
        Loads Gensim Word2Vec weights into the Embedding layers.
        Matches tokens from the SimpleVocab to the W2V vocab.
        """
        print("Loading Pretrained Embeddings...")
        
        def load_w2v(path, vocab, embedding_layer):
            w2v = Word2Vec.load(path)
            hits = 0
            misses = 0
            
            # Iterate over our vocab
            # vocab is a SimpleVocab object
            vocab_dict = vocab.word2idx
            
            with torch.no_grad():
                for token, idx in vocab_dict.items():
                    if token in w2v.wv:
                        vector = torch.from_numpy(w2v.wv[token])
                        # Ensure we don't go out of bounds if vocab grew larger than initialized embedding
                        if idx < embedding_layer.num_embeddings:
                            embedding_layer.weight[idx] = vector
                            hits += 1
                    else:
                        misses += 1
            print(f"Loaded {path}: Hits={hits}, Misses={misses} (Misses are initialized randomly)")

        load_w2v(src_w2v_path, src_vocab, self.src_embed)
        load_w2v(tgt_w2v_path, tgt_vocab, self.tgt_embed)

    def freeze_embeddings(self):
        self.src_embed.weight.requires_grad = False
        self.tgt_embed.weight.requires_grad = False
    def unfreeze_embeddings(self):
        self.src_embed.weight.requires_grad = True
        self.tgt_embed.weight.requires_grad = True

    def encode(self, src, src_mask):
        # Load freqs to device
        freqs_cis = self.freqs_cis[:src.shape[1]].to(src.device)
        
        enc_out = self.src_embed(src) * math.sqrt(self.d_model)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, freqs_cis, src_mask)
        return enc_out

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        # Load freqs to device
        freqs_tgt = self.freqs_cis[:tgt.shape[1]].to(tgt.device)
        
        dec_out = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, freqs_tgt, tgt_mask, src_mask)
            
        logits = self.fc_out(self.final_norm(dec_out))
        return logits

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)

    def create_masks(self, src, tgt, pad_idx_src, pad_idx_tgt):
        # Source mask (PAD mask)
        src_mask = (src != pad_idx_src).unsqueeze(1).unsqueeze(2) # (B, 1, 1, Seq)
        
        # Target mask (Causal + PAD)
        tgt_pad_mask = (tgt != pad_idx_tgt).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        
        tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0)
        
        return src_mask, tgt_mask