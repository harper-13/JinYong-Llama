import math
from dataclasses import dataclass
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore.ops import functional as F

# -----------------------------------------------------------------------------
# ModelArgs

# TODO: 根据需要调整模型参数，可以对比不同参数量的模型的性能


@dataclass
class ModelArgs:
    dim: int = 1024 #origin:1024
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 32 #origin:32
    max_seq_len: int = 2048

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


# -----------------------------------------------------------------------------
# Transformer


class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = ms.Parameter(np.ones(dim, dtype=np.float32))

    def _norm(self, x: ms.Tensor):
        return x * (x.pow(2).mean(-1, keep_dims=True) + self.eps).rsqrt()

    def construct(self, x: ms.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """For RoPE (Rotary Positional Embedding), compute the frequencies of the sinusoids."""
    freqs = 1.0 / (
        theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim)
    )  # (head_dim // 2, )
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
    freqs_cos = np.cos(freqs)
    freqs_sin = np.sin(freqs)
    freqs_cis_real = np.stack([freqs_cos, freqs_sin], axis=-1)
    freqs_cis_real = ms.Tensor(freqs_cis_real, dtype=ms.float32)
    return freqs_cis_real


def apply_rotary_emb(x: ms.Tensor, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.shape[1], 1, xshaped.shape[3], 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = F.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        axis=-1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(start_dim=3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)


def repeat_kv(x: ms.Tensor, n_rep: int) -> ms.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand((bs, slen, n_kv_heads, n_rep, head_dim))
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class KVCache(nn.Cell):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype):
        super(KVCache, self).__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.cache_k = ms.Parameter(
            np.zeros(cache_shape, dtype=dtype), requires_grad=False
        )
        self.cache_v = ms.Parameter(
            np.zeros(cache_shape, dtype=dtype), requires_grad=False
        )

    def update(self, start_pos, xk, xv):
        seqlen = xk.shape[1]
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv


class Attention(nn.Cell):
    def __init__(self, args: ModelArgs):
        super(Attention, self).__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Dense(args.dim, args.n_heads * self.head_dim, has_bias=False)
        self.wk = nn.Dense(args.dim, self.n_kv_heads * self.head_dim, has_bias=False)
        self.wv = nn.Dense(args.dim, self.n_kv_heads * self.head_dim, has_bias=False)
        self.wo = nn.Dense(args.n_heads * self.head_dim, args.dim, has_bias=False)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def construct(
        self,
        x: ms.Tensor,
        start_pos: int,
        freqs_cis: ms.Tensor,
        mask: Optional[ms.Tensor],
    ):
        # TODO: 补全注意力前向传播的代码
        bsz, seqlen, _ = x.shape
        # calculate query, key, value and split out heads
        #new_start
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, n_heads, head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # (bsz, seqlen, n_kv_heads, head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)  # (bsz, seqlen, n_kv_heads, head_dim)
        #new_end
        
        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs(*)
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep) # (bs, cache_len + seqlen, n_, head_dim)
            xv = repeat_kv(xv, self.n_rep) # (bs, cache_len + seqlen, n_, head_dim)
        # make heads be a batch dim(*)
        xq = xq.transpose(0, 2, 1, 3).reshape(bsz * self.n_heads, seqlen, self.head_dim) # (bsz * n_heads, seqlen, head_dim)
        xk = xk.transpose(0, 2, 1, 3).reshape(bsz * self.n_kv_heads, seqlen, self.head_dim) # (bsz * n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(0, 2, 1, 3).reshape(bsz * self.n_kv_heads, seqlen, self.head_dim) # (bsz * n_kv_heads, seqlen, head_dim)

        # attention(*)
        attention_score = F.matmul(xq, xk.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_score += mask
        attention_score = F.softmax(attention_score, axis=-1)
        attention = F.matmul(attention_score, xv)
        # concatenate all the heads
        attention = attention.reshape(bsz, self.n_heads, seqlen, self.head_dim).transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        output = self.wo(attention)
        # output projection
        return output


class FeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super(FeedForward, self).__init__()
        # hidden dim gymnastics that Meta simplified only later
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(hidden_dim, dim, has_bias=False)
        self.w3 = nn.Dense(dim, hidden_dim, has_bias=False)

    def construct(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Cell):
    # TODO: 补全 TransformerBlock 的代码
    # 注意归一化使用 RMSNorm，Attention 的归一化在计算 Attention 之前进行
    # 注意残差连接
    def __init__(self, args: ModelArgs):
        super(TransformerBlock, self).__init__()
        # Attention Layer
        self.attn = Attention(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Feed Forward Layer
        self.ffn = nn.SequentialCell(
            nn.Dense(args.dim, int(args.dim * args.ffn_dim_multiplier) if args.ffn_dim_multiplier else args.dim * 4),
            nn.GELU(),
            nn.Dense(int(args.dim * args.ffn_dim_multiplier) if args.ffn_dim_multiplier else args.dim * 4, args.dim)
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        

    def construct(
        self,
        x: ms.Tensor,
        start_pos: int,
        freqs_cis: ms.Tensor,
        mask: Optional[ms.Tensor],
    ):
        # Attention sublayer with RMSNorm before it
        x_norm = self.attn_norm(x)  # Apply RMSNorm before attention
        attn_output = self.attn(x_norm, start_pos, freqs_cis, mask)  # Attention output
        x = x + attn_output  # Residual connection for Attention
        
        # Feed Forward sublayer with RMSNorm before it
        x_norm = self.ffn_norm(x)  # Apply RMSNorm before feed-forward
        ffn_output = self.feed_forward(x_norm)  # Feed-forward output
        x = x + ffn_output  # Residual connection for Feed Forward
        return x


class Llama(nn.Cell):
    def __init__(self, params: ModelArgs):
        super(Llama, self).__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.logit_layer = nn.Dense(params.dim, self.vocab_size)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.CellList(
            [TransformerBlock(params) for _ in range(params.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Dense(params.dim, params.vocab_size, has_bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def construct(self, tokens: ms.Tensor, start_pos: int):
        # TODO: 补全 Llama 前向传播的代码
        # for use during inference
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = np.full((seqlen, seqlen), float("-inf"))
            mask = np.triu(mask, k=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = np.hstack([np.zeros((seqlen, start_pos)), mask])
            mask = ms.Tensor(mask).type_as(h)

        # Pass through Transformer Blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # Final Normalization
        h = self.norm(h)
        # Output Layer to get logits (logits for vocabulary size)
        logits = self.output(h)
        return logits


    def construct_loss(self, inputs: ms.Tensor, targets: ms.Tensor, ignore_index=-100):
        # TODO: 补全 Llama 前向传播的代码
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        _bsz, seqlen = inputs.shape
        freqs_cis = self.freqs_cis[:seqlen]
        h = self.tok_embeddings(inputs)
        mask = np.full((seqlen, seqlen), float("-inf"), dtype=np.float32)
        mask = np.triu(mask, k=1)
        mask = ms.Tensor(mask, dtype=ms.float32)
        start_pos = -1  # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        logits = self.logit_layer(h)
        fn_loss = nn.SoftmaxCrossEntropyWithLogits(True, "mean")
        logits = logits.reshape(_bsz * seqlen, self.vocab_size)
        targets = targets.reshape(_bsz * seqlen)
        loss = fn_loss(logits,targets)
        return loss




class LlamaWithLoss(nn.Cell):
    """
    Llama training with loss
    """

    def __init__(self, network):
        super(LlamaWithLoss, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, input_ids):
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        loss = self.network.construct_loss(tokens, labels)

        return loss
