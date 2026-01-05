"""
Backup of the encoder-decoder architecture from train-chunni.py.

This file is standalone so you can copy it or move it to another project
without touching your poker training code.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class EncoderSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # NOTE: Removed NANOGPT_SCALE_INIT to give cross-attention stronger signal
        # This is critical - with scale init, cross-attention output is ~0.004 std
        # which gets overwhelmed by self-attention and MLP in residual stream
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Initialize cross-attention to be "louder" so it actually contributes
        self._init_cross_attn_weights()
    
    def _init_cross_attn_weights(self):
        """Initialize cross-attention with larger weights so it's not overwhelmed."""
        # Standard init but with slightly larger std for c_proj
        nn.init.normal_(self.c_attn_q.weight, std=0.02)
        nn.init.zeros_(self.c_attn_q.bias)
        nn.init.normal_(self.c_attn_kv.weight, std=0.02)
        nn.init.zeros_(self.c_attn_kv.bias)
        # c_proj with FULL std=0.02 (not scaled down)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, decoder_x, encoder_output):
        B_dec, T_dec, C = decoder_x.size()
        B_enc, T_enc, C_enc = encoder_output.size()
        assert B_dec == B_enc
        assert C == C_enc

        q = self.c_attn_q(decoder_x)
        kv = self.c_attn_kv(encoder_output)
        k, v = kv.split(self.n_embd, dim=2)
        q = q.view(B_dec, T_dec, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B_enc, T_enc, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B_enc, T_enc, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B_dec, T_dec, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class EncoderSelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = EncoderSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config, causal_self_attention_cls):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = causal_self_attention_cls(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), encoder_output)
        x = x + self.mlp(self.ln_3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [EncoderSelfAttentionBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, config, causal_self_attention_cls):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [DecoderBlock(config, causal_self_attention_cls) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.decoder.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, encoder_idx, decoder_idx, target=None):
        encoder_output = self.encoder(encoder_idx)
        B, T_dec = decoder_idx.size()
        assert T_dec <= self.config.block_size
        pos = torch.arange(0, T_dec, dtype=torch.long, device=decoder_idx.device)
        pos_emb = self.decoder.wpe(pos)
        tok_emb = self.decoder.wte(decoder_idx)
        x = tok_emb + pos_emb
        for block in self.decoder.h:
            x = block(x, encoder_output)
        x = self.decoder.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )
        return logits, loss


