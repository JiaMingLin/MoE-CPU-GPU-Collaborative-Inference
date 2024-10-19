# MoE CPU-GPU Collaborative Inference

## Weights preprocess
1. Weights downloaded from huggingface
```shell
MODEL_PATH="your model path"
# For Mistral 8x7b
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf
# For Phi-3.5
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH --hf --bias
```
2. Weights downloaded from mistral official GitHub page (in tar format)
```shell
MODEL_PATH="your model path"
# For Mistral 8x7b
python3 weights_preprocessor.py --input-path $MODEL_PATH --output-path $MODEL_PATH
```

