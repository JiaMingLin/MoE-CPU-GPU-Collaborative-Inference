# reference: https://github.com/mistralai/mistral-inference
import os
import csv
import nvtx
import time
import argparse
import inspect
import json
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

LOGS = {}  # perf_analysis
EXPERT_CHOICES = {"choice": [], "cnt": []}
CUR_TOKEN_CHOICES = []


def reset_logs():  # perf_analysis
    global LOGS
    global EXPERT_CHOICES
    global CUR_TOKEN_CHOICES
    LOGS = {"attn": [], "ffn_compute": [], "ffn_comm": []}
    EXPERT_CHOICES = {"choice": [], "cnt": []}
    CUR_TOKEN_CHOICES = []
reset_logs()

class CPUMonitor:
    def __init__(self):
        self.running = False
        self.data = []
        self.data_avg = []
        self.OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", 24)
        self.cnt = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_mhz)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def reset(self):
        self.data = []
        self.data_avg = []
        self.cnt = 0

    def get_cpu_temperature(self):
        result = subprocess.run(
            "sensors",
            shell=True,
            capture_output=True,
            text=True
        )
        output = result.stdout.strip().split('\n')
        
        temperatures = {}
        for line in output:
            if 'Tctl' in line or 'Tccd' in line:
                parts = line.split(':')
                label = parts[0].strip()
                temp = parts[1].strip().split(' ')[0].replace('+', '').replace('°C', '')
                temperatures[label] = float(temp)
        
        return temperatures

    def _capture_mhz(self):
        while self.running:
            result = subprocess.run(
                f"cat /proc/cpuinfo | grep 'MHz' | sed 's/cpu MHz[[:space:]]*:[[:space:]]*//' | sort -n | tail -n {self.OMP_NUM_THREADS}",
                shell=True,
                capture_output=True,
                text=True
            )
            # convert to int
            arr = [int(float(x)) for x in result.stdout.strip().split('\n')]
            temp = self.get_cpu_temperature()['Tctl']
            if self.cnt % 30 == 0:
                print([self.cnt, temp] + arr)
            self.data.append([self.cnt, temp] + arr)
            self.data_avg.append([self.cnt, temp, int(mean(arr))])
            self.cnt += 1
            time.sleep(1)

    def save_data(self, path, type="avg"):
        # save data as csv
        data = self.data_avg if type == "avg" else self.data
        title = ["Seconds", "Temperature"] + [f"cpu{i}" for i in range(len(data[0]))]
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(title)
            writer.writerows(data)
    
    def get_data_avg(self, type="avg"):
        return self.data_avg

def dump_expert_choices_to_csv(expert_choices: list, file_path: str):
    """
    Dumps the EXPERT_CHOICES list to a CSV file.

    Args:
        expert_choices (list): The list containing expert choices data.
        file_path (str): The path to the CSV file where expert choices will be dumped.
    """
    # Open the file and write the expert choices
    expert_choices = expert_choices["choice"]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["Index"] + [f"Block_{i//2}" for i in range(64)]
        writer.writerow(header)

        # Write the expert choices
        for i, choice in enumerate(expert_choices):
            row = [i] + choice
            writer.writerow(row)

def dump_expert_cache_to_csv(expert_choices: list, file_path: str, cache_size: int):
    # Open the file and write the expert choices
    expert_choices = expert_choices["cnt"]
    cache_hit_1 = 0
    cache_hit_2 = 0
    total_test = 0
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["Index"] + [f"Block_{i}" for i in range(32)]
        writer.writerow(header)

        # Write the expert choices
        for i, choice in enumerate(expert_choices):
            
            for idx, j in enumerate(choice):
                if idx >= cache_size:
                    break
                if j == 1:
                    cache_hit_1 += 1
                elif j == 2:
                    cache_hit_1 += 1
                    cache_hit_2 += 1
                total_test += 1
            row = [i] + choice
            writer.writerow(row)

    if total_test == 0:
        print("No cache hit due to zero cache size")
    else:
        cache_hit_1 = cache_hit_1 / total_test * 100
        cache_hit_2 = cache_hit_2 / total_test * 100
        print(f"Cache hit 1: {cache_hit_1:.1f}\n Cache hit 2: {cache_hit_2:.1f}")

def dump_token_generation_time_to_csv(logs: list, cpu_freq_avg: list, file_path: str):
    """
    Dumps the LOGS dictionary to a CSV file.

    Args:
        logs (list): The list containing token generation time data.
        file_path (str): The path to the CSV file where token generation time will be dumped.
    """
    # Open the file and write the logs
    with open(file_path, mode='w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Index", "Time", "Timestamp", "Accumulated Generation Throughput (tokens/s)", "CPU Frequency", "CPU Temperature"])

        # Write the log values
        time_sum = 0
        for i, log in enumerate(logs):
            time_sum += log
            timestamp = int(time_sum)
            if timestamp >= len(cpu_freq_avg):
                cpu_freq = cpu_freq_avg[-1]
            else:
                cpu_freq = cpu_freq_avg[timestamp]
            writer.writerow([i+1, log, timestamp, f"{(i+1) / time_sum:.3f}", cpu_freq[-1], cpu_freq[-2]])

def dump_logs_to_csv(logs: dict, file_path: str):
    """
    Dumps the LOGS dictionary to a CSV file.

    Args:
        logs (dict): The dictionary containing log data.
        file_path (str): The path to the CSV file where logs will be dumped.
    """
    # Determine the maximum length of the log lists
    max_length = max(len(values) for values in logs.values())

    # Open the file and write the logs
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Index"] + list(logs.keys()))

        # Write the log values
        for i in range(max_length):
            row = [i]
            for key in logs.keys():
                # Append the value if it exists, otherwise append an empty string
                row.append(logs[key][i] if i < len(logs[key]) else "")
            writer.writerow(row)
            
def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
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


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


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

    @classmethod
    def from_dict(cls, params: dict):
        cls_params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in cls_params})


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


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


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


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

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Retrieve cache
        cache_k = [
            cache_k[:seq_len] for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len] for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

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
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self) -> torch.device:
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

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
        cached_elements = torch.tensor(seqlens, device=self.device, dtype=torch.long)

        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions + batch_idx * self.max_seq_len

        during_prefill = seqpos[0] == 0
        if during_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.max_seq_len
            )
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.max_seq_len)
                .tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=during_prefill,
            mask=mask,
            seqlens=seqlens,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore

class Experts:
    # tmp design:
    # 1. shared across layers
    # 2. weights and computation on CPU

    def __init__(self, ws: dict):
        self.ws = ws

    def init_cache(self, 
                   cache_nblock: int,
                   cache_nway: int,
                   quota: int,
                   replacement_policy: str, # FIFO or LRU
                   device="cuda") -> None:
        self.quota = quota
        self.max_quota = quota
        if cache_nblock != 0:
            single_expert_shape = list(self.ws[f"{0}.{0}"].shape)
            for i in range(32):
                for j in range(8):
                    self.ws[f"{i}.{j}"] = self.ws[f"{i}.{j}"].pin_memory()
            
            self.nblocks = cache_nblock
            self.nways = cache_nway
            self.cache_shape = tuple([cache_nblock, cache_nway] + single_expert_shape)

            self.cache_tag = torch.zeros((cache_nblock, cache_nway), dtype=torch.int8, device="cpu")
            self.cache_valid = torch.zeros((cache_nblock, cache_nway), dtype=torch.int8, device="cpu")
            print(f"Cache shape: {self.cache_shape}")

            self.replacement_policy = replacement_policy

            self.cache_FIFO = []
            self.cache_LRU = []
            self.cuda_stream = torch.cuda.Stream(device="cuda")
            self.cache_device = device
            with torch.cuda.stream(self.cuda_stream):
                self.cache = torch.empty(self.cache_shape, dtype=torch.bfloat16, device=self.cache_device)
            for i in range(self.nblocks):
                push_list_FIFO = []
                push_list_LRU = []
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
            self.cache_valid[li][i] = 1

    def reset_quota(self) -> None:
        self.quota = self.max_quota

    def cache_aware_forward(self, li: int, ei_list: list, x: torch.Tensor) -> torch.Tensor:
        exp1 = ei_list[0]
        exp2 = ei_list[1]
        ret_l = []
        if li < self.nblocks:
            cache_hit = [self.check_cache_hit(li, exp1), self.check_cache_hit(li, exp2)]
            if cache_hit[0] != -1 and cache_hit[1] != -1:
                for i in range(2):
                    # self.check_data_cpy_finished(li, cache_hit[i])
                    w = self.cache[li, cache_hit[i]]
                    ret_l.append((nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
            else:
                x_cpu = x.to("cpu")

                # self.check_data_cpy_finished(li, cache_hit[i])
                for i in range(2):
                    # CPU compute first, and move expert to GPU
                    if cache_hit[i] == -1:
                        w = self.ws[f"{li}.{ei_list[i]}"]
                        with nvtx.annotate(f"block{li} CPU compute expert", color="orange"):
                            ret_l.append((nn.functional.silu(x_cpu @ w[0].T) * (x_cpu @ w[2].T)) @ w[1].to())
                        self.quota -= 1
                        if self.quota > 0:
                            with torch.cuda.stream(self.cuda_stream):
                                dst_way = self.pop_cache(li)
                                self.cache_tag[li, dst_way] = ei_list[i]
                                self.cache_valid[li, dst_way] = 2
                                self.cache[li, dst_way].copy_(self.ws[f"{li}.{ei_list[i]}"], non_blocking=True)
                                    # self.cache[li, dst_way] = self.ws[f"{li}.{ei_list[i]}"].to(self.cache[li, dst_way].device, non_blocking=True)
                                    # w.to(self.cache[li, dst_way], non_blocking=True)
                    else:
                        w = self.cache[li, cache_hit[i]]
                        ret_l.append((nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
            return ret_l, cache_hit
        else:                                                           # cache miss (due to insufficient cache size)
            x_cpu = x.to("cpu")
            w1 = self.ws[f"{li}.{ei_list[0]}"]
            w2 = self.ws[f"{li}.{ei_list[1]}"]
            ret_l.append((nn.functional.silu(x_cpu @ w1[0].T) * (x_cpu @ w1[2].T)) @ w1[1])
            ret_l.append((nn.functional.silu(x_cpu @ w2[0].T) * (x_cpu @ w2[2].T)) @ w2[1])
            return ret_l, [-1, -1]

    def forward(self, li: int, ei: int, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        w = self.ws[f"{li}.{ei}"]
        
        st = time.time()
        # force GPU, used for motivational purposes
        if (device == "gpu"):       
            w = w.cuda()

        ed = time.time()
        ret = (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]  # type: ignore
        ed2 = time.time()

        weight_2_gpu = ed - st
        ffn = ed2 - ed
        return ret, weight_2_gpu, ffn


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.li = li
        self.gate = gate
        self.experts = experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global EXPERT_CHOICES
        global CUR_TOKEN_CHOICES
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        mode = 3
        if results.shape[0] == 1:
            if mode == 0:
                st = time.time()
                ex = inputs[0].to("cpu")
                ed = time.time()
                token_2_cpu = ed - st
                ey = []
                ey_device = []

                st = time.time()
                for i in selected_experts[0]:
                    a, _, _ = self.experts.forward(self.li, i, ex)
                    ey.append(a)
                ed = time.time()
                ffn_compute = ed - st

                st = time.time()
                for i in range(2):
                    ey_device.append(ey[i].to(weights.device))
                ed = time.time()
                token_2_gpu = ed - st

                for i in range(2):
                    batch_idx, nth_expert = torch.where(selected_experts == selected_experts[0][i])
                    results[0] += (weights[0, nth_expert, None] * ey_device[i])[0]
                
                if selected_experts[0][0] > selected_experts[0][1]:
                    CUR_TOKEN_CHOICES.append(selected_experts[0][1].item())
                    CUR_TOKEN_CHOICES.append(selected_experts[0][0].item())
                else:
                    CUR_TOKEN_CHOICES.append(selected_experts[0][0].item())
                    CUR_TOKEN_CHOICES.append(selected_experts[0][1].item())
                # print(CUR_TOKEN_CHOICES)

                # print(token_2_cpu*1000, ffn_compute*1000, token_2_gpu*1000)
                return results, token_2_cpu+token_2_gpu, ffn_compute, 0
            elif mode == 2:
                ex = inputs[0]
                expert_list = []
                
                for i in selected_experts[0]:
                    ey, weight_2_gpu, ffn_compute = self.experts.forward(self.li, i, ex, device='gpu')
                    batch_idx, nth_expert = torch.where(selected_experts == i)
                    results[0] += (weights[0, nth_expert, None] * ey)[0]
                weight_2_gpu = weight_2_gpu * 2
                ffn_compute = ffn_compute * 2

                # print(weight_2_gpu*1000, ffn_compute*1000)
                return results, weight_2_gpu, ffn_compute, 0
            elif mode == 3:     # cache-aware
                ex = inputs[0]
                st = time.time()
                ret_l, cache_hit = self.experts.cache_aware_forward(self.li, selected_experts[0].tolist(), ex)

                # if selected_experts[0][0] > selected_experts[0][1]:
                #     CUR_TOKEN_CHOICES.append(selected_experts[0][1].item())
                #     CUR_TOKEN_CHOICES.append(selected_experts[0][0].item())
                # else:
                #     CUR_TOKEN_CHOICES.append(selected_experts[0][0].item())
                #     CUR_TOKEN_CHOICES.append(selected_experts[0][1].item())

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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, li: int, experts: Experts):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        st = time.time()
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        ed = time.time()
        atten_time = ed - st
        r, ffn_comm_time, ffn_compute_time, cache_hit_cnt = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out, atten_time, ffn_comm_time, ffn_compute_time, cache_hit_cnt


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.args = args
        self._precomputed_freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(args=args, li=li, experts=experts)
                for li in range(args.n_layers)
            }
        )
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
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )

        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        global EXPERT_CHOICES
        global CUR_TOKEN_CHOICES

        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]
        
        atten_time_list = []
        ffn_comm_list = []
        ffn_compute_list = []
        cache_hit_count_list = []
        
        for li in range(self.args.n_layers):
            cache_view = cache.get_view(li, input_metadata)
            with nvtx.annotate(f"block{li}", color="red"):
                h, atten_time, ffn_comm, ffn_compute, cache_hit_count = self.layers[str(li)](h, freqs_cis, cache_view)
            atten_time_list.append(atten_time)
            ffn_comm_list.append(ffn_comm)
            ffn_compute_list.append(ffn_compute)
            cache_hit_count_list.append(cache_hit_count)

        self.experts.reset_quota()
        atten_time_avg = torch.mean(torch.tensor(atten_time_list), dtype=float).item()*1000
        ffn_comm_avg = torch.mean(torch.tensor(ffn_comm_list), dtype=float).item()*1000
        ffn_compute_avg = torch.mean(torch.tensor(ffn_compute_list), dtype=float).item()*1000

        LOGS["attn"].append(atten_time_avg)
        LOGS["ffn_comm"].append(ffn_comm_avg)
        LOGS["ffn_compute"].append(ffn_compute_avg)
        
        EXPERT_CHOICES["choice"].append(CUR_TOKEN_CHOICES.copy())
        EXPERT_CHOICES["cnt"].append(cache_hit_count_list.copy())
        CUR_TOKEN_CHOICES = []

        cache.update_seqlens(seqlens)
        outs = self.output(self.norm(h))
        return outs.float()
    
    def init_cache(self, cache_nblock: int, cache_nway: int, quota: int, replacement_policy: str) -> None:
        self.experts.init_cache(cache_nblock, cache_nway, quota, replacement_policy=replacement_policy)

    @staticmethod
    def load(model_path: Path, gpu: torch.device) -> "Transformer":
        model_args = ModelArgs.from_dict(get_json(model_path / "params.json"))

        non_experts = torch.load(
            model_path / "non-experts.pt",
            map_location=gpu,
            mmap=True,
        )
        experts = torch.load(
            model_path / "experts.pt", map_location=torch.device("cpu"), mmap=True
        )
        exp = Experts(experts)
        
        
        with torch.device("meta"):
            model = Transformer(args=model_args, experts=exp)
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: MistralTokenizer,
    model: Transformer,
    gpu: torch.device,
    *,
    max_tokens: int,
    max_batch_size: int = 64,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
    profile: bool = False,
) -> Tuple[List[str], int, float, int, float]:
    model = model.eval()
    prefill_tic = torch.cuda.Event(enable_timing=True)
    prefill_toc = torch.cuda.Event(enable_timing=True)
    prefill_tic.record()

    encoded_prompts: List[List[int]] = [
        tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=p)])
        ).tokens
        for p in prompts
    ]
    B, V = len(encoded_prompts), model.args.vocab_size
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.args.n_layers,
        max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # prefill / prompt evaluation stage
    prelogits = model.forward(
        torch.tensor(sum(encoded_prompts, []), device=model.device, dtype=torch.long),
        seqlens=seqlens,
        cache=cache,
    )
    last_positions = torch.tensor(seqlens, device=prelogits.device).cumsum(dim=0) - 1
    last_token_prelogits = prelogits.index_select(0, last_positions)
    prefill_toc.record()
    torch.cuda.synchronize(device=gpu)
    prefill_time = prefill_tic.elapsed_time(prefill_toc) / 1000  # to seconds

    # decode
    decode_tic = torch.cuda.Event(enable_timing=True)
    decode_toc = torch.cuda.Event(enable_timing=True)
    decode_tic.record()
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    if profile is True:
        cpumonitor = CPUMonitor()
        cpumonitor.start()
    token_gen_time_history = []
    prev_token_time = time.time()
    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)
        is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        token_gen_time_history.append(time.time() - prev_token_time)
        prev_token_time = time.time()
        assert last_token_prelogits.shape == (B, V)

    generated_tokens: List[List[int]]
    n_gen_tkns = 0
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
        n_gen_tkns = sum(len(y) - 1 for y in generated_tokens)
    else:
        generated_tokens = []
    responses = [tokenizer.decode(y) for y in generated_tokens]
    decode_toc.record()
    torch.cuda.synchronize(device=gpu)
    decode_time = decode_tic.elapsed_time(decode_toc) / 1000  # to seconds
    if profile is True:
        cpumonitor.stop()
        cpumonitor.save_data("cpu_freq_avg.csv")
        dump_token_generation_time_to_csv(token_gen_time_history, cpumonitor.get_data_avg(), "token_gen_time.csv")

    return (
        seqlens,
        responses,
        sum(seqlens),
        prefill_time,
        n_gen_tkns,
        decode_time,
    )


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def main(
    model_path: str,
    prompt: str,
    prompt_path: str,
    n_prompts: int,
    max_tokens: int,
    hide_resp: bool,
    cache_nblocks: int,
    cache_nways: int,
    cache_quota: int,
    csv_report_file: str,
    cache_report_file: str,
    cache_replace_policy: str
):
    global EXPERT_CHOICES
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    gpu_0 = torch.device("cuda:0")
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]
    tokenizer = MistralTokenizer.v1()
    model = Transformer.load(Path(model_path), gpu_0)

    model.init_cache(cache_nblocks, cache_nways, cache_quota, cache_replace_policy)

    # warmup
    generate(
        ["hello, how are you?"],
        tokenizer,
        model,
        gpu_0,
        max_tokens=1,
        max_batch_size=len(prompts),
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    reset_logs()

    seqlens, responses, n_p_tkns, prefill_time, n_gen_tkns, decode_time = generate(
        prompts,
        tokenizer,
        model,
        gpu_0,
        max_tokens=max_tokens,
        max_batch_size=len(prompts),
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        profile=True,
    )
    print("=" * 20)
    print("PERFORMANCE BREAKDOWN\n")
    print("PROMPT EVALUATION:")
    print(f"token count: {n_p_tkns}")
    print(f"total time in sec(s): {prefill_time:.2f}")
    print(f"throughput: {(n_p_tkns / prefill_time):.2f} t/s")
    print("TOKEN GENERATION:")
    print(f"token count: {n_gen_tkns}")
    print(f"total time in sec(s): {decode_time:.2f}")
    if n_gen_tkns > 0:
        print(f"throughput: {(n_gen_tkns / decode_time):.2f} t/s")
    else:
        responses = ["" for _ in prompts]
    if not hide_resp:
        print("=" * 20)
        print("In-n-Outs")
        print(f"AVG seqlen: {mean(seqlens)}")
        print(f"seqlens: {seqlens}\n")
        for p, resp in zip(prompts, responses):
            print(f"PROMPT:\n{p}")
            print(f"RESPONSE:\n{resp}\n")

    dump_logs_to_csv(LOGS, csv_report_file)
    dump_expert_choices_to_csv(EXPERT_CHOICES, "expert_choices.csv")
    dump_expert_cache_to_csv(EXPERT_CHOICES, cache_report_file, cache_nblocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--hide-resp", action="store_true")
    parser.add_argument("--cache-nblocks", type=int, default=0)
    parser.add_argument("--cache-nways", type=int, default=0)
    parser.add_argument("--cache-quota", type=int, default=64)
    parser.add_argument("--breakdown-csv", type=str, default="out.csv")
    parser.add_argument("--cachehit-csv", type=str, default="cache.csv")
    parser.add_argument("--cache-replace-policy", type=str, default="FIFO")
    args = parser.parse_args()

    main(
        args.model_path,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.max_tokens,
        args.hide_resp,
        args.cache_nblocks,
        args.cache_nways,
        args.cache_quota,
        args.breakdown_csv,
        args.cachehit_csv,
        args.cache_replace_policy
    )
