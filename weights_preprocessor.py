from pathlib import Path
import argparse
import gc
import json
import logging

import safetensors.torch

import torch

class WeightsPreprocessor:

    def __init__(self, input_path: str, output_path: str, hf: bool) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.hf = hf
        self.config = None
        # self.attention_bias = attention_bias
        # self.lm_head_bias

    def get_hf_model_configs(self):
        try:
            with open(self.input_path / "config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

    def _torch_load_cpu(self, path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_hf_weights(self) -> dict:
        safetensors_index = self.input_path / "model.safetensors.index.json"
        if safetensors_index.exists():
            with open(safetensors_index, "r") as f:
                metadata = json.load(f)
            ws = {}
            for filename in set(metadata["weight_map"].values()):
                ws.update(
                    safetensors.torch.load_file(self.input_path / filename,
                                                device="cpu"))
            return ws

        pytorch_index = self.input_path / "pytorch_model.bin.index.json"
        if pytorch_index.exists():
            with open(pytorch_index, "r") as f:
                metadata = json.load(f)
            ws = {}
            for filename in set(metadata["weight_map"].values()):
                ws.update(self._torch_load_cpu(self.input_path / filename))
            return ws

        pytorch_single = self.input_path / "pytorch_model.bin"
        if pytorch_single.exists():
            return self._torch_load_cpu(pytorch_single)

        logging.error(
            "No supported HF weights found. Expected one of: "
            "model.safetensors.index.json, pytorch_model.bin.index.json, pytorch_model.bin "
            f"in {self.input_path}")
        raise FileNotFoundError(
            f"No supported HF weights found in {self.input_path}")

    def _get_first_present(self, source: dict, keys: list[str], *, default=None):
        for key in keys:
            if key in source:
                return source[key]
        if default is not None:
            return default
        raise KeyError(f"None of keys found: {keys}")

    def _get_model_architecture(self) -> str:
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        # Fallback to model_type when architectures is missing.
        return self.config.get("model_type", "")

    def _pop_first_present(self, source: dict, keys: list[str]):
        for key in keys:
            if key in source:
                return source.pop(key)
        raise KeyError(f"None of keys found: {keys}")

    def process_hf_config(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_arch = self._get_model_architecture()
        supported_architectures = {"MixtralForCausalLM", "MiniCPMMoEForCausalLM"}
        if model_arch and model_arch not in supported_architectures:
            logging.warning(
                f"Unexpected architecture '{model_arch}'. Trying Mixtral-compatible parsing."
            )

        conf = {}
        conf["dim"] = self._get_first_present(self.config, ["hidden_size"])
        conf["n_layers"] = self._get_first_present(self.config,
                                                   ["num_hidden_layers"])
        conf["n_heads"] = self._get_first_present(self.config,
                                                  ["num_attention_heads"])
        conf["head_dim"] = conf["dim"] // conf["n_heads"]
        conf["hidden_dim"] = self._get_first_present(self.config,
                                                     ["intermediate_size"])
        conf["n_kv_heads"] = self._get_first_present(
            self.config, ["num_key_value_heads", "num_kv_heads"],
            default=conf["n_heads"])
        conf["norm_eps"] = self._get_first_present(self.config,
                                                   ["rms_norm_eps"])
        conf["vocab_size"] = self._get_first_present(self.config, ["vocab_size"])
        conf["rope_theta"] = self._get_first_present(self.config, ["rope_theta"],
                                                     default=10000.0)
        conf["moe"] = {
            "num_experts_per_tok": self._get_first_present(
                self.config, ["num_experts_per_tok"]),
            "num_experts":
            self._get_first_present(self.config,
                                    ["num_local_experts", "num_experts"]),
        }
        if self.config.get("attention_bias", False):
            conf["attention_bias"] = True
            self.attention_bias = True
        else:
            self.attention_bias = False
        if self.config.get("lm_head_bias", False):
            conf["lm_head_bias"] = True
            self.lm_head_bias = True
        else:
            self.lm_head_bias = False
        rope_scaling = self.config.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            conf["rope_scaling"] = {
                "short_factor": rope_scaling["short_factor"],
                "mscale": rope_scaling["short_mscale"],
            }
        conf["max_position_embeddings"] = self._get_first_present(
            self.config, ["max_position_embeddings"])
        conf["scale_emb"] = self._get_first_present(self.config, ["scale_emb"],
                                                    default=1.0)
        conf["scale_depth"] = self._get_first_present(self.config,
                                                      ["scale_depth"],
                                                      default=1.0)
        conf["model_type"] = model_arch
        with open(self.output_path / "params.json", "w") as f:
            json.dump(conf, f)

    def process_hf_experts(self, ws: dict) -> None:
        experts = {}
        n_layers = self._get_first_present(self.config, ["num_hidden_layers"])
        n_experts = self._get_first_present(self.config,
                                            ["num_local_experts", "num_experts"])
        for li in range(n_layers):
            for ei in range(n_experts):
                # Keep Mixtral-compatible mapping for MiniCPM-MoE experts.
                w1 = self._pop_first_present(
                    ws, [
                        f"model.layers.{li}.block_sparse_moe.experts.{ei}.w1.weight",
                        f"model.layers.{li}.mlp.experts.{ei}.w1.weight",
                    ])
                w2 = self._pop_first_present(
                    ws, [
                        f"model.layers.{li}.block_sparse_moe.experts.{ei}.w2.weight",
                        f"model.layers.{li}.mlp.experts.{ei}.w2.weight",
                    ])
                w3 = self._pop_first_present(
                    ws, [
                        f"model.layers.{li}.block_sparse_moe.experts.{ei}.w3.weight",
                        f"model.layers.{li}.mlp.experts.{ei}.w3.weight",
                    ])
                experts[f"{li}.{ei}"] = torch.stack((w1, w2.T, w3), dim=0)
            gc.collect()

        torch.save(experts, self.output_path / "experts.pt")
        logging.info("finished processing expert weights")

        return ws

    def process_hf_non_experts(self, ws: dict) -> None:
        tok_embeddings = ws.pop("model.embed_tokens.weight")
        if "lm_head.weight" in ws:
            output_weight = ws.pop("lm_head.weight")
        else:
            # Some MiniCPM checkpoints tie output projection with token embeddings.
            logging.warning(
                "lm_head.weight not found, using model.embed_tokens.weight as output.weight"
            )
            output_weight = tok_embeddings

        non_experts = {
            "tok_embeddings.weight": tok_embeddings,
            "norm.weight": ws.pop("model.norm.weight"),
            "output.weight": output_weight,
        }
        if self.lm_head_bias:
            non_experts["norm.bias"] = ws.pop("model.norm.bias")
            non_experts["output.bias"] = ws.pop("lm_head.bias")
        for li in range(self.config["num_hidden_layers"]):
            prefix = f"model.layers.{li}"
            pfx = prefix[6:]
            non_experts[f"{pfx}.attention_norm.weight"] = ws.pop(
                f"{prefix}.input_layernorm.weight")
            non_experts[f"{pfx}.ffn_norm.weight"] = ws.pop(
                f"{prefix}.post_attention_layernorm.weight")
            if self.attention_bias:
                non_experts[f"{pfx}.attention_norm.bias"] = ws.pop(
                    f"{prefix}.input_layernorm.bias")
                non_experts[f"{pfx}.ffn_norm.bias"] = ws.pop(
                    f"{prefix}.post_attention_layernorm.bias")
            non_experts[f"{pfx}.feed_forward.gate.weight"] = self._pop_first_present(
                ws, [
                    f"{prefix}.block_sparse_moe.gate.weight",
                    f"{prefix}.mlp.gate.weight",
                ])
            for pi in ["q", "k", "v", "o"]:
                non_experts[f"{pfx}.attention.w{pi}.weight"] = ws.pop(
                    f"{prefix}.self_attn.{pi}_proj.weight")
                if self.attention_bias:
                    non_experts[f"{pfx}.attention.w{pi}.bias"] = ws.pop(
                        f"{prefix}.self_attn.{pi}_proj.bias")

        # safetensors.torch.save_file(
        #     non_experts, self.output_path / f"non-experts.safetensors"
        # )
        torch.save(non_experts, self.output_path / "non-experts.pt")
        logging.info("finished processing non-expert weights")

        return ws

    def get_pth_model_configs(self):
        try:
            with open(self.input_path / "params.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

    def load_pth_weights(self) -> dict:
        return torch.load(self.input_path / "consolidated.00.pth", mmap=True)

    def process_pth_experts(self, ws: dict) -> None:
        experts = {}
        for li in range(self.config["n_layers"]):
            for ei in range(self.config["moe"]["num_experts"]):
                w1 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w1.weight")
                w2 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w2.weight")
                w3 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w3.weight")
                experts[f"{li}.{ei}"] = torch.stack((w1, w2.T, w3), dim=0)

        torch.save(experts, self.output_path / "experts.pt")
        logging.info("finished processing expert weights")

        return ws

    def process_pth_non_experts(self, ws: dict) -> None:
        torch.save(ws, self.output_path / f"non-experts.pt")
        logging.info("finished processing non-expert weights")
        return ws

    def start(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.hf:
            self.config = self.get_hf_model_configs()
            self.process_hf_config()
            ws = self.process_hf_experts(self.load_hf_weights())
            gc.collect()
            ws = self.process_hf_non_experts(ws)
            print(ws)
        else:
            self.config = self.get_pth_model_configs()
            ws = self.process_pth_experts(self.load_pth_weights())
            self.process_pth_non_experts(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--hf",
                        action="store_true")  # uses pth weights by default
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_preprocessor = WeightsPreprocessor(args.input_path,
                                               args.output_path, args.hf)
    weights_preprocessor.start()
