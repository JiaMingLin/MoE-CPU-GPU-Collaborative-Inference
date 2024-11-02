# reference: https://github.com/mistralai/mistral-inference
import json
import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias, BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)

from monitoring import LOGS, EXPERT_CHOICES, CUR_TOKEN_CHOICES
import nvtx

MIXTRAL_MODEL_TYPE = "MixtralForCausalLM"
PHI_MODEL_TYPE = "PhiMoEForCausalLM"


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    moe: dict
    max_position_embeddings: int = 128_000
    lm_head_bias: bool = False
    attention_bias: bool = False
    rope_scaling: dict = None
    model_type: str = MIXTRAL_MODEL_TYPE

    @classmethod
    def from_dict(cls, params: dict):
        cls_params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in cls_params})


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int],
                     device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(positions=torch.cat(
            [torch.arange(0, seqlen)
             for seqlen in seqlens]).to(device=device, dtype=torch.long))


@dataclass
class CacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


class CacheView:

    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk)
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv)

    def interleave_kv(self, xk: torch.Tensor,
                      xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(
            xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(
            xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Retrieve cache
        cache_k = [
            cache_k[:seq_len]
            for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len]
            for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]

        def interleave_list(l1: List[torch.Tensor],
                            l2: List[torch.Tensor]) -> List[torch.Tensor]:
            assert len(l1) == len(l2)
            return [v for pair in zip(l1, l2) for v in pair]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim))
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int,
                 metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id],
                         metadata, self.kv_seqlens)

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros((batch_size, ),
                                      device=self.device,
                                      dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens,
                                        device=self.device,
                                        dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        """
        Get metadata about cache positions
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert isinstance(self.kv_seqlens, torch.Tensor)
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        cached_elements = torch.tensor(seqlens,
                                       device=self.device,
                                       dtype=torch.long)

        positions = torch.cat([
            torch.arange(pos, pos + seqlen)
            for pos, seqlen in zip(seqpos, seqlens)
        ]).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions + batch_idx * self.max_seq_len

        during_prefill = seqpos[0] == 0
        if during_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(
                seqlens).make_local_attention(self.max_seq_len)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(
                    max=self.max_seq_len).tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=during_prefill,
            mask=mask,
            seqlens=seqlens,
        )


# reference: microsoft/Phi-3.5-MoE-instruct: modeling_phimoe.py
class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):

    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.short_factor = config.rope_scaling["short_factor"]
        self.short_mscale = config.rope_scaling["mscale"]

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        rescale_factors = torch.tensor(self.short_factor,
                                       dtype=torch.float32,
                                       device=x.device)
        mscale = self.short_mscale
        assert rescale_factors.shape == (self.dim // 2, ), \
            f"misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}"

        inv_freq = 1.0 / (rescale_factors * (self.base**(
            torch.arange(0, self.dim, 2).float().to(x.device) / self.dim)))

        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(x.dtype), (emb.sin() * mscale).to(
            x.dtype)


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        bias = args.attention_bias
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=bias)
        self.wk = nn.Linear(args.dim,
                            args.n_kv_heads * args.head_dim,
                            bias=bias)
        self.wv = nn.Linear(args.dim,
                            args.n_kv_heads * args.head_dim,
                            bias=bias)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=bias)

        if args.model_type == PHI_MODEL_TYPE:
            self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                self.head_dim, self.args)

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self,
                              q,
                              k,
                              cos,
                              sin,
                              position_ids=None,
                              unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:, None, :]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)

        if self.args.model_type == PHI_MODEL_TYPE:
            kv_seq_len = xk.shape[0] + (0 if cache is None else
                                        cache.key.shape[1])
            cos, sin = self.rotary_emb(xv, seq_len=kv_seq_len)
            xq, xk = self._apply_rotary_pos_emb(xq,
                                                xk,
                                                cos,
                                                sin,
                                                position_ids=positions)
        else:  # MixtralForCausalLM
            xq, xk = self._apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads,
                           self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads,
                           self.head_dim)

        def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int,
                      dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
            keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
            values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
            return keys, values

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]

        # xq = xq.to(dtype=torch.float16)
        # key = key.to(dtype=torch.float16)
        # val = val.to(dtype=torch.float16)
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask)
        # output = output.to(dtype=torch.bfloat16)

        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore


class Experts:
    # tmp design:
    # 1. shared across layers
    # 2. weights and computation on CPU

    def __init__(self, model_args: ModelArgs, ws: dict):
        self.ws = ws
        self.args = model_args

    def init_cache(
            self,
            cache_nblock: int,
            cache_nway: int,
            quota: int,
            replacement_policy: str,  # FIFO or LRU
            device="cuda") -> None:
        single_expert_shape = list(self.ws[f"{0}.{0}"].shape)
        for i in range(self.args.n_layers):
            for j in range(self.args.moe["num_experts"]):
                self.ws[f"{i}.{j}"] = self.ws[f"{i}.{j}"].pin_memory()
        self.quota = quota
        self.max_quota = quota

        self.nblocks = cache_nblock
        self.nways = cache_nway
        self.cache_shape = tuple([cache_nblock, cache_nway] +
                                 single_expert_shape)

        self.cache_tag = torch.zeros((cache_nblock, cache_nway),
                                     dtype=torch.int8,
                                     device="cpu")
        self.cache_valid = torch.zeros((cache_nblock, cache_nway),
                                       dtype=torch.int8,
                                       device="cpu")
        self.cache_valid_cuda_event = []
        print(f"Cache shape: {self.cache_shape}")

        self.replacement_policy = replacement_policy

        self.cache_FIFO = []
        self.cache_LRU = []
        self.cuda_stream = torch.cuda.Stream(device="cuda")
        self.cache_device = device
        with torch.cuda.stream(self.cuda_stream):
            self.cache = torch.empty(self.cache_shape,
                                     dtype=torch.bfloat16,
                                     device=self.cache_device)
        for i in range(self.nblocks):
            push_list_FIFO = []
            push_list_LRU = []
            self.cache_valid_cuda_event.append([None] * self.nways)
            for j in range(self.nways):
                push_list_FIFO.append(j)
                push_list_LRU.append(0)
            self.cache_FIFO.append(push_list_FIFO)
            self.cache_LRU.append(push_list_LRU)

    def pop_cache(self, li: int) -> None:
        if self.replacement_policy == "FIFO":
            x = self.cache_FIFO[li].pop(0)
            self.cache_FIFO[li].append(x)
        else:
            # get argmin from self.cache_LRU[li]
            argmin_index = self.cache_LRU[li].index(min(self.cache_LRU[li]))
            x = argmin_index
            mx = max(self.cache_LRU[li])
            self.cache_LRU[li][argmin_index] = mx + 1
        return x

    def check_cache_hit(self, li: int, ei: int) -> bool:
        for i in range(self.nways):
            if self.cache_tag[li, i] == ei and self.cache_valid[li, i] > 0:
                self.cache_LRU[li][i] += 1
                return i
        return -1

    def check_data_cpy_finished(self, li: int, i: int) -> bool:
        if self.cache_valid[li][i] == 2:
            # self.cuda_stream[li].synchronize()
            self.cache_valid_cuda_event[li][i].synchronize()
            self.cache_valid[li][i] = 1

    def reset_quota(self) -> None:
        self.quota = self.max_quota

    def cache_aware_forward(self, li: int, ei_list: list,
                            x: torch.Tensor) -> torch.Tensor:
        exp1 = ei_list[0]
        exp2 = ei_list[1]
        ret_l = []
        if li < self.nblocks:
            cache_hit = [
                self.check_cache_hit(li, exp1),
                self.check_cache_hit(li, exp2)
            ]
            if cache_hit[0] != -1 and cache_hit[1] != -1:
                for i in range(2):
                    self.check_data_cpy_finished(li, cache_hit[i])
                    w = self.cache[li, cache_hit[i]]
                    ret_l.append(
                        (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
            else:
                x_cpu = x.to("cpu")

                # self.check_data_cpy_finished(li, cache_hit[i])
                for i in [0, 1]:
                    # CPU compute first, and move expert to GPU
                    if cache_hit[i] == -1:
                        w = self.ws[f"{li}.{ei_list[i]}"]
                        with nvtx.annotate(f"block{li} CPU compute expert",
                                           color="orange"):
                            ret_l.append((nn.functional.silu(x_cpu @ w[0].T) *
                                          (x_cpu @ w[2].T)) @ w[1].to())
                        self.quota -= 1
                        if self.quota > 0:
                            with torch.cuda.stream(self.cuda_stream):
                                dst_way = self.pop_cache(li)
                                self.cache_tag[li, dst_way] = ei_list[i]
                                self.cache_valid[li, dst_way] = 2
                                self.cache[li, dst_way].copy_(
                                    self.ws[f"{li}.{ei_list[i]}"],
                                    non_blocking=True)
                                self.cache_valid_cuda_event[li][
                                    dst_way] = torch.cuda.Event()
                                self.cache_valid_cuda_event[li][
                                    dst_way].record(self.cuda_stream)
                                # self.cache[li, dst_way] = self.ws[f"{li}.{ei_list[i]}"].to(self.cache[li, dst_way].device, non_blocking=True)
                                # w.to(self.cache[li, dst_way], non_blocking=True)
                    else:
                        self.check_data_cpy_finished(li, cache_hit[i])
                        w = self.cache[li, cache_hit[i]]

                        ret_l.append((nn.functional.silu(x @ w[0].T) *
                                      (x @ w[2].T)) @ w[1])
            return ret_l, cache_hit
        else:  # cache miss (due to insufficient cache size)
            x_cpu = x.to("cpu")
            w1 = self.ws[f"{li}.{ei_list[0]}"]
            w2 = self.ws[f"{li}.{ei_list[1]}"]
            ret_l.append((nn.functional.silu(x_cpu @ w1[0].T) *
                          (x_cpu @ w1[2].T)) @ w1[1])
            ret_l.append((nn.functional.silu(x_cpu @ w2[0].T) *
                          (x_cpu @ w2[2].T)) @ w2[1])
            return ret_l, [-1, -1]

    def forward(self,
                li: int,
                ei: int,
                x: torch.Tensor,
                device: str = "cpu") -> torch.Tensor:
        w = self.ws[f"{li}.{ei}"]

        st = time.time()
        # force GPU, used for motivational purposes
        if (device == "gpu"):
            w = w.cuda()

        ed = time.time()
        ret = (nn.functional.silu(x @ w[0].T) *
               (x @ w[2].T)) @ w[1]  # type: ignore
        ed2 = time.time()

        weight_2_gpu = ed - st
        ffn = ed2 - ed
        return ret, weight_2_gpu, ffn


class MoeLayer(nn.Module):

    def __init__(self, args: ModelArgs, li: int, gate: nn.Module,
                 experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        if "sparsemixer" in args.moe:
            self.use_sparsemixer = True
            self.router_jitter_noise = args.moe['sparsemixer'][
                'router_jitter_noise']
        else:
            self.use_sparsemixer = False
        self.li = li
        self.gate = gate
        self.experts = experts

    def sparsemixer(self, scores, jitter_eps, top_k=2):
        """
        Sparse mixer function to select top-k experts and compute multipliers.
        Based on the paper: https://arxiv.org/pdf/2409.12136
        We first replace the TopK(·) function as random sampling of discrete variables
        in model training. Then, following Liu et al. (2023a) and Liu et al. (2023b), we apply Heun's
        third order method to approximate the expert routing gradient and construct a modified
        back-propagation to give a mathematically sound gradient estimation for expert routing.

        Args:
            scores (torch.Tensor): Input scores tensor.
            jitter_eps (float): Jitter epsilon for numerical stability.
            top_k (int): Number of top experts to select.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Multiplier and selected experts tensors.
        """
        if top_k != 2:
            raise ValueError("top_k must be equal to 2")

        # first expert

        with torch.no_grad():
            # Compute mask for sparsity
            mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = ((mask_logits_threshold - scores) /
                                     factor) > (2 * jitter_eps)

        # Apply mask
        masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
        selected_experts = max_ind

        # Compute scores for gradients
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

        multiplier = multiplier_o

        # Masked out first expert
        masked_scores = torch.scatter(
            scores,
            -1,
            selected_experts,
            float("-inf"),
        )
        with torch.no_grad():
            # Compute mask for sparsity
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1,
                                                               keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = ((mask_logits_threshold - scores) /
                                     factor) > (2 * jitter_eps)

        # Apply mask
        masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold,
                                                      float("-inf"))
        selected_experts_top2 = max_ind
        # Compute scores for gradients
        masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
        multiplier_top2_o = masked_gates_top2.gather(
            dim=-1, index=selected_experts_top2)

        multiplier_top2 = multiplier_top2_o

        multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
        selected_experts = torch.concat(
            (selected_experts, selected_experts_top2), dim=-1)

        return (
            multiplier,
            selected_experts,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global EXPERT_CHOICES
        global CUR_TOKEN_CHOICES
        gate_logits = self.gate(inputs)
        if self.use_sparsemixer:
            weights, selected_experts = self.sparsemixer(
                gate_logits,
                jitter_eps=self.router_jitter_noise,
            )
        else:
            weights, selected_experts = torch.topk(gate_logits,
                                                   self.num_experts_per_tok)
            weights = F.softmax(weights, dim=1,
                                dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        if results.shape[0] == 1:
            ex = inputs[0]
            st = time.time()
            ret_l, cache_hit = self.experts.cache_aware_forward(
                self.li, selected_experts[0].tolist(), ex)
            
            CUR_TOKEN_CHOICES.extend(
                sorted([selected_experts[0][0].item(), selected_experts[0][1].item()]))

            for idx, i in enumerate(selected_experts[0]):
                batch_idx, nth_expert = torch.where(selected_experts == i)
                ret_l[idx] = ret_l[idx].to(weights.device)
                results[0] += (weights[0, nth_expert, None] * ret_l[idx])[0]
            ed = time.time()
            if cache_hit[0] == -1 and cache_hit[1] == -1:
                hit_count = 0
            elif cache_hit[0] == -1 or cache_hit[1] == -1:
                hit_count = 1
            else:
                hit_count = 2
            ffn_compute = ed - st
            return results, 0, ffn_compute, hit_count
        else:
            for ei in range(self.num_experts):
                batch_idx, nth_expert = torch.where(selected_experts == ei)
                ex = inputs[batch_idx].to("cpu")
                ey, _, _ = self.experts.forward(self.li, ei, ex)
                ey = ey.to(weights.device)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * ey
        return results, 0, 0, 0


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        output = output * self.weight
        if self.bias is not None:
            output += self.bias
        return output


class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs, li: int, experts: Experts):
        super().__init__()
        self.attention = Attention(args)
        bias = args.attention_bias if args.model_type == PHI_MODEL_TYPE else False
        # Mistral 8x7b
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, bias=bias)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, bias=bias)
        # Phi-3.5-MoE
        # self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=bias, elementwise_affine=True)
        # self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=bias, elementwise_affine=True)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                positions: torch.Tensor,
                cache: Optional[CacheView]) -> torch.Tensor:
        st = time.time()
        r = self.attention.forward(self.attention_norm(x), freqs_cis,
                                   positions, cache)
        h = x + r
        ed = time.time()
        atten_time = ed - st
        r, ffn_comm_time, ffn_compute_time, cache_hit_cnt = self.feed_forward.forward(
            self.ffn_norm(h))
        out = h + r
        return out, atten_time, ffn_comm_time, ffn_compute_time, cache_hit_cnt


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.args = args
        self._precomputed_freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        lm_bias = args.lm_head_bias if args.model_type == PHI_MODEL_TYPE else False
        self.norm = RMSNorm(args.dim, eps=args.norm_eps, bias=lm_bias)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=lm_bias)
        self.layers = nn.ModuleDict({
            str(li):
            TransformerBlock(args=args, li=li, experts=experts)
            for li in range(args.n_layers)
        })
        self.experts = experts

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # default to 10**6
            theta = self.args.rope_theta or 1000000.0
            self._precomputed_freqs_cis = self._precompute_freqs_cis(
                self.args.head_dim, 128_000, theta)

        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device)
        return self._precomputed_freqs_cis

    def _precompute_freqs_cis(self, dim: int, end: int,
                              theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta
                       **(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        global EXPERT_CHOICES
        global CUR_TOKEN_CHOICES

        (num_toks, ) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        if self.args.model_type == PHI_MODEL_TYPE:
            freqs_cis = torch.tensor(0)
        else:  # MixtralForCausalLM
            freqs_cis = self.freqs_cis[input_metadata.positions]

        atten_time_list = []
        ffn_comm_list = []
        ffn_compute_list = []
        cache_hit_count_list = []

        for li in range(self.args.n_layers):
            cache_view = cache.get_view(li, input_metadata)
            with nvtx.annotate(f"block{li}", color="red"):
                h, atten_time, ffn_comm, ffn_compute, cache_hit_count = self.layers[
                    str(li)](h, freqs_cis, input_metadata.positions,
                             cache_view)
            atten_time_list.append(atten_time)
            ffn_comm_list.append(ffn_comm)
            ffn_compute_list.append(ffn_compute)
            cache_hit_count_list.append(cache_hit_count)

        self.experts.reset_quota()
        atten_time_avg = torch.mean(torch.tensor(atten_time_list),
                                    dtype=float).item() * 1000
        ffn_comm_avg = torch.mean(torch.tensor(ffn_comm_list),
                                  dtype=float).item() * 1000
        ffn_compute_avg = torch.mean(torch.tensor(ffn_compute_list),
                                     dtype=float).item() * 1000

        LOGS["attn"].append(atten_time_avg)
        LOGS["ffn_comm"].append(ffn_comm_avg)
        LOGS["ffn_compute"].append(ffn_compute_avg)

        EXPERT_CHOICES["choice"].append(CUR_TOKEN_CHOICES.copy())
        EXPERT_CHOICES["cnt"].append(cache_hit_count_list.copy())
        CUR_TOKEN_CHOICES = []

        cache.update_seqlens(seqlens)
        outs = self.output(self.norm(h))
        return outs.float()

    def init_cache(self, cache_nblock: int, cache_nway: int, quota: int,
                   replacement_policy: str) -> None:
        self.experts.init_cache(cache_nblock,
                                cache_nway,
                                quota,
                                replacement_policy=replacement_policy)

    @staticmethod
    def load(model_path: Path, gpu: torch.device) -> "Transformer":
        with open(model_path / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))

        non_experts = torch.load(
            model_path / "non-experts.pt",
            weights_only=model_args.model_type == PHI_MODEL_TYPE,
            map_location=gpu,
            mmap=True,
        )
        experts = torch.load(
            model_path / "experts.pt",
            weights_only=model_args.model_type == PHI_MODEL_TYPE,
            map_location=torch.device("cpu"),
            mmap=True)
        exp = Experts(model_args, experts)

        with torch.device("meta"):
            model = Transformer(args=model_args, experts=exp)
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model
