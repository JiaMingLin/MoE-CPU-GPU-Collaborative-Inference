# MoE CPU-GPU Collaborative Inference

## Weights preprocess
1. Weights need to be downloaded from huggingface
2. Processes the weights and model configuration
```shell
MODEL_PATH="your model path"
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf
```

## Execution

```bash
python3 src/main.py --model-path "your model path"
python3 src/main.py --cache-nblocks {block size} --cache-nways {# of blocks} --cache-replace-policy {LRU/FIFO}
```
