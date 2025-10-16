# Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems

[![Paper](https://img.shields.io/badge/Paper-Accepted-green)](ASP-DAC26_CPU_GPU_MoE.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

**Official implementation of "Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems"**

*En-Ming Huang¹, Li-Shang Lin², Chun-Yi Lee¹†*  
¹National Taiwan University, ²National Tsing Hua University  
†Corresponding author

This repository contains the official implementation of our research accepted at **ASP-DAC 2026**. Our framework addresses the memory constraints of consumer-grade hardware by strategically leveraging both CPU and GPU resources, incorporating intelligent expert caching mechanisms to achieve superior inference performance.

## Research Overview

Large Language Models (LLMs) have achieved impressive results across various tasks, yet their high computational demands pose deployment challenges, especially on consumer-grade hardware. Mixture of Experts (MoE) models provide an efficient solution through selective activation of parameter subsets, reducing computation requirements. However, state-of-the-art MoE models still require substantial memory beyond typical consumer GPU capacities.

Our novel CPU-GPU collaborative inference framework addresses these challenges through:

- **Expert Caching on GPU**: N-index, M-way set-associative cache structure for frequently accessed experts
- **Asynchronous Cache Miss Handling**: CPU computation during cache misses with background expert fetching
- **Dynamic Expert Reuse Patterns**: Exploits consecutive layer and token patterns for optimal caching
- **Zero Model Modification**: Compatible with existing MoE architectures without requiring changes

## Key Features & Contributions

- **CPU-GPU Collaborative Framework**: Efficiently leverages CPU multi-core parallelism for expert computations while using GPU memory as intelligent cache
- **Expert Reuse Pattern Exploitation**: Identifies and leverages consecutive layer (44% reuse) and consecutive token (40-60% reuse) patterns
- **Asynchronous Expert Caching**: N-index, M-way set-associative cache with LRU eviction policy and background expert fetching
- **Superior Performance**: Achieves up to **4.8 tokens/s** (Mixtral 8x7B) and **10.4 tokens/s** (Phi-3.5-MoE) with **4.4× speedup** over prefetching methods
- **Energy Efficiency**: Reduces energy consumption to **29.9%** of prefetching methods while maximizing hardware utilization
- **Zero Modification Required**: Works out-of-the-box with existing MoE models without architecture changes or dataset profiling

## Requirements

- **Hardware**: Consumer-grade GPU (RTX 3080/4090 series), Multi-core CPU (8+ cores recommended)
- **Software**: Python 3.8+, PyTorch 2.0+, CUDA 11.8+
- **Memory**: GPU with 16GB+ VRAM and CPU with 128GB+ main memory for optimal performance

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference.git
cd MoE-CPU-GPU-Collaborative-Inference
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers safetensors mistral-common xformers nvtx-plugins
```

## Supported Models

Our framework supports state-of-the-art MoE architectures validated in our research:

| Model | Architecture | Layers | Experts/Layer | Top-K | Expert Size | Total Size |
|-------|-------------|---------|---------------|-------|-------------|------------|
| **Mixtral 8x7B** | MixtralForCausalLM | 32 | 8 | 2 | 340 MB | 88 GB |
| **Phi-3.5-MoE** | PhiMoEForCausalLM | 32 | 16 | 2 | 152 MB | 79 GB |

> **Note**: The framework is designed to work with any MoE architecture following the standard router-expert pattern without requiring model modifications.

## Model Setup

### 1. Download Model Weights

Download your preferred MoE model from Hugging Face:

```bash
# Example: Mixtral-8x7B-Instruct-v0.1
MODEL_PATH="/path/to/your/model"
git clone https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 $MODEL_PATH
```

### 2. Preprocess Weights

Convert and optimize the model weights for our framework:

```bash
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf
```

**Parameters:**
- `--input-path`: Path to the downloaded model directory
- `--output-path`: Output directory for processed weights
- `--hf`: Enable Hugging Face format processing

## Usage

### Basic Inference

Run inference with default settings:

```bash
python3 src/main.py --model-path "/path/to/your/model" --prompt "Explain quantum computing"
```

### Advanced Configuration

Leverage CPU-GPU collaborative inference with expert caching:

```bash
# Optimal configuration for RTX 4090 (24GB) with Mixtral 8x7B
python3 src/main.py \
    --model-path "/path/to/your/model" \
    --prompt "Your prompt here" \
    --cache-nblocks 14 \
    --cache-nways 4 \
    --cache-replace-policy LRU \
    --max-tokens 512

# High-performance setup with 24 CPU cores
export OMP_NUM_THREADS=24
python3 src/main.py \
    --model-path "/path/to/your/model" \
    --prompt "Explain quantum computing in detail" \
    --cache-nblocks 14 \
    --cache-nways 4 \
    --max-tokens 1024
```

### Batch Processing

Process multiple prompts from a file:

```bash
python3 src/main.py \
    --model-path "/path/to/your/model" \
    --prompt-path "prompts.txt" \
    --n-prompts 100 \
    --max-tokens 256
```

## Configuration Options

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--model-path` | Path to the model directory | Required | - |
| `--prompt` | Single prompt for inference | None | Any string |
| `--prompt-path` | File containing multiple prompts | None | Text file path |
| `--n-prompts` | Number of prompts to process | None | Integer |
| `--max-tokens` | Maximum tokens to generate | Required | 256-1024 |
| `--cache-nblocks` | Number of cache indexes (N) | 0 | 14 (RTX 4090 + Mixtral) |
| `--cache-nways` | Cache associativity (M) | 0 | 2-4 (fewer cores), 4-8 (more cores) |
| `--cache-quota` | Cache quota per expert | 64 | 64 |
| `--cache-replace-policy` | Cache replacement policy | FIFO | **LRU** (recommended) |
| `--breakdown-csv` | Performance breakdown output | out.csv | File path |
| `--cachehit-csv` | Cache hit rate output | cache.csv | File path |
| `--hide-resp` | Hide response output | False | Flag |

### Cache Configuration Guidelines

**Cache Sizing Formula**: `Total Slots (S) = Available GPU Memory / Expert Size`
- **Mixtral 8x7B on RTX 4090**: ~56 slots → 14 indexes × 4 ways
- **Phi-3.5-MoE on RTX 4090**: ~128 slots → 16 indexes × 8 ways

**CPU Core Optimization**:
- **Low cores (1-4)**: Use more indexes, fewer ways → maximize coverage
- **High cores (8-24)**: Use fewer indexes, more ways → minimize transfers

## Performance Analysis & Monitoring

Our framework provides comprehensive performance analysis capabilities:

### **Expert Selection Pattern Analysis**
- **Consecutive Layers Pattern**: ~44% expert reuse across consecutive layers
- **Consecutive Tokens Pattern**: 40-60% expert reuse across consecutive tokens
- **Extended Reuse**: 23% share experts with previous 2 tokens, 18% with 3+ tokens

### **Cache Performance Metrics**
- **Hit Rate Analysis**: "expert(s) hit" vs "2 experts hit" tracking
- **LRU vs FIFO vs Random**: LRU shows 5-15% improvement over random selection
- **Cache Configuration Impact**: Optimal balance between coverage and hit rate

### **Real-time Performance Monitoring**
- **Token Generation Speed**: Per-token timing and throughput analysis
- **CPU Frequency Scaling**: Monitors frequency changes with core count (4.8-5.3 GHz)
- **Power Consumption**: CPU package power via RAPL, GPU power via nvidia-smi
- **Memory Transfer Analysis**: PCIe Gen 4.0 ×16 bandwidth utilization

### **Detailed Profiling Outputs**
- `expert_choices.csv`: Expert selection patterns per layer/token
- `token_gen_time.csv`: Per-token generation timing with CPU frequency data
- `cpu_freq_avg.csv`: CPU frequency scaling analysis
- Performance breakdown CSV with cache hit/miss statistics

## Experimental Results

### **Performance Achievements**

| Model | Our Method | Pre-gated MoE | Speedup | Energy Reduction |
|-------|------------|---------------|---------|------------------|
| **Mixtral 8x7B** | **4.8 tokens/s** | 1.1 tokens/s | **4.4×** | **70.1%** |
| **Phi-3.5-MoE** | **10.4 tokens/s** | 2.4 tokens/s | **4.3×** | **72.2%** |

### **CPU Core Scaling Analysis**

| CPU Cores | Mixtral 8x7B (tokens/s) | Phi-3.5-MoE (tokens/s) | Power (W) |
|-----------|-------------------------|------------------------|-----------|
| 2 cores   | 2.1                     | 4.2                    | 91.7      |
| 8 cores   | 3.8                     | 8.1                    | 111.0     |
| 24 cores  | **4.8**                 | **10.4**               | 147.5     |


## Research Applications & Extensions

This codebase enables various research directions and has been validated on consumer-grade hardware:

### **Core Research Areas**
1. **Expert Caching Strategies**: LRU vs. FIFO vs. custom replacement policies
2. **CPU-GPU Load Balancing**: Optimal resource allocation across different hardware configurations
3. **Memory Hierarchy Optimization**: Multi-level caching and prefetching strategies
4. **Power-Performance Trade-offs**: Energy-efficient inference on resource-constrained systems
5. **Expert Selection Pattern Mining**: Advanced pattern recognition for improved caching

### **Hardware Compatibility Studies**
- **Consumer GPUs**: RTX 3080/3090/4080/4090 series validation
- **CPU Architectures**: AMD Threadripper, Intel Core series scaling analysis  
- **Memory Bandwidth**: PCIe Gen 3.0/4.0/5.0 impact studies


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Huang2026CPUGPUMOE,
    title={Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems},
    author={Huang, En-Ming and Lin, Li-Shang and Lee, Chun-Yi},
    booktitle={Proc. Asia and South Pacific Design Automation Conf. (ASP-DAC)},
    year={2026},
    note={Available at: https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference}
}
```

### **Related Publications**
This work builds upon and extends research in:
- Mixture of Experts architectures and sparse computation
- CPU-GPU heterogeneous computing for deep learning
- Memory-efficient inference on consumer hardware
- Expert caching and prefetching strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was supported by:
- **National Science and Technology Council (NSTC)**, Taiwan under grants:
  - NSTC 114-2221-E-002-069-MY3
  - NSTC 113-2221-E-002-212-MY3  
  - NSTC 114-2218-E-A49-026
  - NSTC 114-2640-E-002-006
- **Google Inc.** for research support
- **National Center for High-Performance Computing** for computational resources

### **Technical Acknowledgments**
- [Mistral AI](https://mistral.ai/) for the Mixtral model architecture and reference implementation
- [Microsoft](https://www.microsoft.com/) for the Phi-3.5-MoE model
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [xFormers](https://github.com/facebookresearch/xformers) for efficient attention implementations
- [PyTorch](https://pytorch.org/) and [OpenMP](https://www.openmp.org/) for multi-threading support
