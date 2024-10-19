# MoE CPU-GPU Collaborative Inference

## Weights preprocess
1. Weights downloaded from huggingface
```shell
MODEL_PATH="your model path"
# For Mistral 8x7b
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf
# For Phi-3.5
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf
```
2. Weights downloaded from mistral official GitHub page (in tar format)
```shell
MODEL_PATH="your model path"
# For Mistral 8x7b
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH
```

## Execution
There are two types of implementation: `moe_hf.py` and `mixtral.py`. The main difference is that `moe_hf.py` uses the tokenizer library from huggingface, and therefore supports both Mixtral 8x7b and Phi-3.5-MoE. On the other hand, the `mixtral.py` is modified from the [mistral-inference](https://github.com/mistralai/mistral-inference/tree/main), which employed the tokenizer made by themselves.

### Models downloaded from huggingface
```bash
python3 src/moe_hf.py --model-path "your model path"
```

### Mistral Weights downloaded from mistral official GitHub page (in tar format)
```bash
python3 src/mixtral.py  --model-path "your model path"
```
