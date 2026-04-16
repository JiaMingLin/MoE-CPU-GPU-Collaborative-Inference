# 2026-04-16 MiniCPM-MoE 除錯與修改紀錄

## 背景

今天目標是讓 `MiniCPM-MoE-8x2b` 能透過本專案流程完成：

1. 權重預處理（`weights_preprocessor.py`）
2. 推論執行（`src/main.py`）

過程中依序排除了 config 相容性、權重格式相容性、tokenizer 相容性與執行參數問題。

---

## 修改摘要

### `weights_preprocessor.py`

- 新增 HF 權重載入相容性：
  - 支援 `model.safetensors.index.json`
  - 支援 `pytorch_model.bin.index.json`
  - 支援單檔 `pytorch_model.bin`
- 新增 `_torch_load_cpu()`，兼容不同 PyTorch 版本（`weights_only=True` 不可用時 fallback）。
- 修正 `rope_scaling` 為 `null` 時的崩潰：
  - 僅在 `rope_scaling` 為 `dict` 時才取子欄位。
- 在流程開始與寫 config 前都確保 `output_path` 存在（自動 `mkdir`）。
- 新增 `_pop_first_present()`，用於多種 key 命名的 fallback 取值。
- 對齊 MiniCPM 權重命名：
  - experts 支援 `block_sparse_moe.experts.*` 與 `mlp.experts.*`
  - gate 支援 `block_sparse_moe.gate.weight` 與 `mlp.gate.weight`
- 修正 `lm_head.weight` 缺失時（tied embedding）：
  - fallback 使用 `model.embed_tokens.weight` 作為 `output.weight`。

### `src/main.py`

- 擴增 HF tokenizer 模型型別判斷：
  - `PhiMoEForCausalLM`
  - `MiniCPMForCausalLM`
  - `MiniCPMMoEForCausalLM`
- 新增 `--tokenizer-path` 參數，允許 processed model 與原始 tokenizer 分離路徑。
- `AutoTokenizer.from_pretrained()` 改為：
  - `trust_remote_code=True`
  - `use_fast=False`
  - `fix_mistral_regex=True`
- `--max-tokens` 增加預設值 `128`，並在 `main()` 補 `None` 防呆。
- 調整 HF prompt/tokenize 路徑：
  - 優先使用 `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`
  - 若不可用則 fallback `encode(...)`
  - 修正 `apply_chat_template()` 可能回傳 `BatchEncoding/dict/list` 的型別差異，統一轉為 `List[int]`
- 不再強制手動包 `<|user|>...<|assistant|>` 字串，改由 tokenizer template 主導。

### `src/monitoring.py`

- 先前為避開 `Tctl` 例外曾加容錯 fallback（`_pick_temperature` 等），後續依需求已改回較嚴格原邏輯：
  - 固定讀 `Tctl`
  - 移除 fallback temperature path
  - `save_data()` 與 token-time CSV 也回到原先行為

---

## 已安裝套件（環境：`aspdac26`）

- `sentencepiece`
- `protobuf`

---

## 建議執行指令

### 1) 權重預處理

```bash
python weights_preprocessor.py \
  --input-path /home/jmlin/models/MiniCPM-MoE-8x2b \
  --output-path /home/jmlin/models/processed/MiniCPM-MoE-8x2b \
  --hf
```

### 2) 推論執行

```bash
python3 src/main.py \
  --model-path "/home/jmlin/models/processed/MiniCPM-MoE-8x2b" \
  --tokenizer-path "/home/jmlin/models/MiniCPM-MoE-8x2b" \
  --prompt "Explain quantum computing" \
  --max-tokens 128
```

---

## 目前狀態與注意事項

- 程式可執行到完整生成流程，性能 breakdown 可輸出。
- 若輸出內容仍偏亂碼/標點，下一步建議檢查：
  - `params.json` 與模型架構欄位是否完全對齊執行端預期
  - `experts.pt` / `non-experts.pt` 的權重映射是否與 `src/model.py` 嚴格一致
  - 先用已知可用 Mixtral 權重做 A/B 排除 tokenizer 以外因素

