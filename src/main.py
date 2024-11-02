import time
import argparse
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple, Union

import torch

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from transformers import AutoTokenizer

from monitoring import (
    dump_expert_choices_to_csv,
    dump_expert_cache_to_csv,
    dump_token_generation_time_to_csv,
    dump_logs_to_csv,
    LOGS, EXPERT_CHOICES,
    CPUMonitor,
    reset_logs,
)

from model import Transformer, BufferCache


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: Union[MistralTokenizer, AutoTokenizer],
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

    if model.args.model_type == "PhiMoEForCausalLM":
        encoded_prompts: List[List[int]] = [
            tokenizer.encode(p, add_special_tokens=True) for p in prompts
        ]
    else: # MixtralForCausalLM
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
        torch.tensor(sum(encoded_prompts, []),
                     device=model.device,
                     dtype=torch.long),
        seqlens=seqlens,
        cache=cache,
    )
    last_positions = torch.tensor(seqlens,
                                  device=prelogits.device).cumsum(dim=0) - 1
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
        next_token = sample(last_token_prelogits,
                            temperature=temperature,
                            top_p=0.8)
        is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token,
                                             seqlens=[1] * B,
                                             cache=cache)
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
        dump_token_generation_time_to_csv(token_gen_time_history,
                                          cpumonitor.get_data_avg(),
                                          "token_gen_time.csv")

    return (
        seqlens,
        responses,
        sum(seqlens),
        prefill_time,
        n_gen_tkns,
        decode_time,
    )


def sample(logits: torch.Tensor, temperature: float,
           top_p: float) -> torch.Tensor:
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

def main(model_path: str, prompt: str, prompt_path: str, n_prompts: int,
         max_tokens: int, hide_resp: bool, cache_nblocks: int,
         cache_nways: int, cache_quota: int, csv_report_file: str,
         cache_report_file: str, cache_replace_policy: str):
    global EXPERT_CHOICES
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)

    gpu_0 = torch.device("cuda:0")
    model = Transformer.load(Path(model_path), gpu_0)
    if model.args.model_type == "PhiMoEForCausalLM":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else: # MixtralForCausalLM
        tokenizer = MistralTokenizer.v1()

    
    prompts: list[str] = None
    if prompt:
        if model.args.model_type == "PhiMoEForCausalLM":
            prompts = [f"<|user|>{prompt}<|end|><|assistant|>"]
        else: # MixtralForCausalLM
            prompts = [f"{prompt}"]
    else:
        with open(Path(prompt_path), "r") as f:
            dataset: list[str] = f["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]
    
    model.init_cache(cache_nblocks, cache_nways, cache_quota,
                     cache_replace_policy)

    eos_token_id = tokenizer.eos_token_id if model.args.model_type == "PhiMoEForCausalLM" else tokenizer.instruct_tokenizer.tokenizer.eos_id
    # warmup
    generate(
        ["hello, how are you?"],
        tokenizer,
        model,
        gpu_0,
        max_tokens=1,
        max_batch_size=len(prompts),
        eos_id=eos_token_id,
    )
    reset_logs()

    seqlens, responses, n_p_tkns, prefill_time, n_gen_tkns, decode_time = generate(
        prompts,
        tokenizer,
        model,
        gpu_0,
        max_tokens=max_tokens,
        max_batch_size=len(prompts),
        eos_id=eos_token_id,
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

    reset_logs()
    main(args.model_path, args.prompt, args.prompt_path, args.n_prompts,
         args.max_tokens, args.hide_resp, args.cache_nblocks, args.cache_nways,
         args.cache_quota, args.breakdown_csv, args.cachehit_csv,
         args.cache_replace_policy)