#!/usr/bin/env python3
"""Create Hugging Face DFlash sources and deterministic GPU-FP16 PA fixtures.

`tiny-source` writes a normal Transformers `model.safetensors` checkpoint.
OpenVINO export is deliberately external through `optimum-cli export openvino`.
The reference commands never execute an OpenVINO SDPA model: the C++ simulator
owns SDPAToPagedAttention and GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config


SIMULATOR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SIMULATOR_ROOT.parent
TARGET_ROOT = SIMULATOR_ROOT / "target"
OPTIMUM_ROOT = PROJECT_ROOT / "optimum-intel"
if str(OPTIMUM_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMUM_ROOT))

from optimum.exporters.openvino.__main__ import update_config_for_custom_draft_model


DEFAULT_REAL_MODEL_ID = "z-lab/Qwen3.5-9B-DFlash"


@dataclass(frozen=True)
class Case:
    name: str
    block_len: int
    delta_lens: tuple[int, int]


CASES = {
    "tiny": Case("tiny", block_len=4, delta_lens=(15, 3)),
    "real": Case("real", block_len=6, delta_lens=(15, 3)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_xpu() -> torch.device:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "This fixture generator requires a PyTorch XPU build and an available XPU. "
            "It intentionally has no CPU/CUDA fallback."
        )
    return torch.device("xpu")


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=8,
    )
    config.architectures = ["DFlashDraftModel"]
    config.dflash_config = {
        "target_layer_ids": [0, 1],
        "mask_token_id": 1,
        "block_size": CASES["tiny"].block_len,
    }
    # Qwen3Config normalizes these to regular Qwen defaults during __init__;
    # restore the DFlash mixed-attention configuration explicitly.
    config.layer_types = ["full_attention", "sliding_attention"]
    config.sliding_window = 8
    config.use_sliding_window = True
    config.max_window_layers = 2
    config.use_cache = True
    config._attn_implementation = "sdpa"
    return config


def require_dflash_config(config: Any, source: str) -> None:
    architectures = getattr(config, "architectures", None) or []
    if not architectures or architectures[0] != "DFlashDraftModel":
        raise ValueError(f"{source} is not a DFlashDraftModel source model")
    if not getattr(config, "dflash_config", {}).get("target_layer_ids"):
        raise ValueError(f"{source} has no DFlash target-layer metadata")


def create_tiny_source_model(output: Path, seed: int) -> None:
    """Create a regular Hugging Face safetensors checkpoint, not an OpenVINO IR."""
    seed_torch(seed)
    config = tiny_config()
    update_config_for_custom_draft_model(config, "Qwen3DFlashDraftModel", "Qwen3DFlashForCausalLM")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval().to(dtype=torch.float16)
    model.config.architectures = ["DFlashDraftModel"]
    model.save_pretrained(output, safe_serialization=True)
    # PreTrainedModel.save_pretrained records the concrete implementation
    # class as `architectures`; restore the DFlash export dispatch marker.
    model.config.architectures = ["DFlashDraftModel"]
    model.config.layer_types = ["full_attention", "sliding_attention"]
    model.config.sliding_window = 8
    model.config.use_sliding_window = True
    model.config.max_window_layers = 2
    model.config.save_pretrained(output)
    if not (output / "model.safetensors").is_file():
        raise RuntimeError("Tiny source model did not produce model.safetensors")


def load_dflash_model(model_id_or_path: str | Path, revision: str | None = None) -> Any:
    revision_kwargs = {"revision": revision} if revision else {}
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True, **revision_kwargs)
    require_dflash_config(config, str(model_id_or_path))
    update_config_for_custom_draft_model(config, "Qwen3DFlashDraftModel", "Qwen3DFlashForCausalLM")
    return AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **revision_kwargs,
    ).eval()


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    mapping = {
        torch.float16: "f16",
        torch.float32: "f32",
        torch.int32: "i32",
        torch.int64: "i64",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported fixture tensor dtype: {tensor.dtype}") from error


class FixtureWriter:
    def __init__(self, root: Path):
        self.root = root
        self.tensor_dir = root / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = ["version=1"]

    def set(self, key: str, value: str | int) -> None:
        self.lines.append(f"{key}={value}")

    def tensor(self, name: str, tensor: torch.Tensor) -> None:
        host = tensor.detach().to("cpu").contiguous()
        dtype = tensor_dtype_name(host)
        array = host.numpy()
        filename = f"{name.replace('.', '_')}.bin"
        path = self.tensor_dir / filename
        array.tofile(path)
        shape = ",".join(str(value) for value in array.shape)
        self.lines.append(f"tensor={name}|{dtype}|{shape}|tensors/{filename}")

    def write(self) -> None:
        (self.root / "manifest.txt").write_text("\n".join(self.lines) + "\n")


def cache_tensors(cache: Any, layer_count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index in range(layer_count):
        layer = cache.layers[layer_index]
        result.append((layer.keys, layer.values))
    return result


def make_step_inputs(
    model: Any,
    device: torch.device,
    past_len: int,
    delta_len: int,
    block_len: int,
) -> dict[str, torch.Tensor]:
    config = model.config
    target_layer_ids = config.dflash_config["target_layer_ids"]
    target_hidden_width = len(target_layer_ids) * config.hidden_size
    return {
        "inputs_embeds": torch.randn(
            1, block_len, config.hidden_size, device=device, dtype=torch.float16
        ),
        "hidden_states": torch.randn(
            1, delta_len, target_hidden_width, device=device, dtype=torch.float16
        ),
        "position_ids": torch.arange(
            past_len, past_len + delta_len + block_len, device=device, dtype=torch.int64
        ).unsqueeze(0),
        "attention_mask": torch.ones(
            1, past_len + delta_len + block_len, device=device, dtype=torch.int64
        ),
        "token_type_ids": torch.ones(1, block_len, device=device, dtype=torch.int64),
    }


def add_model_metadata(writer: FixtureWriter, model: Any, model_dir: Path, case: Case) -> None:
    config = model.config
    model_xml = model_dir / "openvino_model.xml"
    model_bin = model_dir / "openvino_model.bin"
    model_config = model_dir / "config.json"
    if not model_xml.is_file() or not model_bin.is_file() or not model_config.is_file():
        raise FileNotFoundError(f"Expected OpenVINO model files under {model_dir}")

    writer.set("case", case.name)
    writer.set("steps", 2)
    writer.set("model_xml_sha256", sha256(model_xml))
    writer.set("model_bin_sha256", sha256(model_bin))
    writer.set("model_config_sha256", sha256(model_config))
    writer.set("model_layers", config.num_hidden_layers)
    writer.set("hidden_size", config.hidden_size)
    writer.set("target_hidden_width", len(config.dflash_config["target_layer_ids"]) * config.hidden_size)
    writer.set("kv_heads", config.num_key_value_heads)
    writer.set("head_dim", config.head_dim)
    writer.set("weight_precision", "f16")
    writer.set("reference_device", "xpu")
    writer.set("reference_precision", "f16")
    writer.set("block_len", case.block_len)


@torch.inference_mode()
def generate_reference(
    model: Any,
    model_dir: Path,
    output: Path,
    case: Case,
    seed: int,
    model_id: str,
    source_revision: str,
) -> None:
    device = require_xpu()
    seed_torch(seed)
    model = model.to(device=device, dtype=torch.float16).eval()
    torch.xpu.synchronize()

    writer = FixtureWriter(output)
    writer.set("source_model", model_id)
    writer.set("source_revision", source_revision)
    writer.set("seed", seed)
    add_model_metadata(writer, model, model_dir, case)

    past_key_values = None
    committed_len = 0
    for step, delta_len in enumerate(case.delta_lens):
        inputs = make_step_inputs(model, device, committed_len, delta_len, case.block_len)
        output_state = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        torch.xpu.synchronize()

        full_len = committed_len + delta_len + case.block_len
        next_committed_len = committed_len + delta_len
        if output_state.last_hidden_state.shape[1] != case.block_len - 1:
            raise AssertionError(
                f"Unexpected draft output length: {output_state.last_hidden_state.shape[1]}, "
                f"expected {case.block_len - 1}"
            )
        if output_state.past_key_values.get_seq_length() != full_len:
            raise AssertionError(
                f"Unexpected full cache length: {output_state.past_key_values.get_seq_length()}, "
                f"expected {full_len}"
            )

        writer.set(f"step.{step}.past_len", committed_len)
        writer.set(f"step.{step}.delta_len", delta_len)
        writer.set(f"step.{step}.block_len", case.block_len)
        writer.set(f"step.{step}.full_len", full_len)
        writer.set(f"step.{step}.committed_len", next_committed_len)
        for name, tensor in inputs.items():
            writer.tensor(f"step{step}.{name}", tensor)
        writer.tensor(f"step{step}.expected.last_hidden_state", output_state.last_hidden_state.to(torch.float16))

        layer_count = model.config.num_hidden_layers
        for layer_index, (key, value) in enumerate(cache_tensors(output_state.past_key_values, layer_count)):
            if key.shape[2] != full_len or value.shape[2] != full_len:
                raise AssertionError(f"Unexpected cache length in layer {layer_index}")
            writer.tensor(f"step{step}.expected.key.{layer_index}", key.to(torch.float16))
            writer.tensor(f"step{step}.expected.value.{layer_index}", value.to(torch.float16))

        # The PA runner will retain only the target-hidden delta and release the
        # seed-plus-proposal B-row suffix after validation.
        output_state.past_key_values.crop(next_committed_len)
        if output_state.past_key_values.get_seq_length() != next_committed_len:
            raise AssertionError("DFlash proposal-tail crop did not retain the committed prefix")
        for layer_index, (key, value) in enumerate(cache_tensors(output_state.past_key_values, layer_count)):
            writer.tensor(f"step{step}.committed.key.{layer_index}", key.to(torch.float16))
            writer.tensor(f"step{step}.committed.value.{layer_index}", value.to(torch.float16))

        past_key_values = output_state.past_key_values
        committed_len = next_committed_len

    writer.write()


def add_output_arguments(parser: argparse.ArgumentParser, default: Path) -> None:
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tiny_source = commands.add_parser(
        "tiny-source",
        help="create a tiny Hugging Face DFlash safetensors checkpoint",
    )
    add_output_arguments(tiny_source, TARGET_ROOT / "tiny" / "source")
    tiny_source.add_argument("--seed", type=int, default=20260806)

    tiny_reference = commands.add_parser(
        "tiny-reference",
        help="generate XPU reference tensors from an already exported tiny source model",
    )
    add_output_arguments(tiny_reference, TARGET_ROOT / "tiny" / "reference")
    tiny_reference.add_argument("--source-model-dir", type=Path, default=TARGET_ROOT / "tiny" / "source")
    tiny_reference.add_argument("--openvino-model-dir", type=Path, default=TARGET_ROOT / "tiny" / "model")
    tiny_reference.add_argument("--seed", type=int, default=20260806)

    real_reference = commands.add_parser(
        "real-reference",
        help="generate XPU reference tensors for the real DFlash checkpoint",
    )
    add_output_arguments(real_reference, TARGET_ROOT / "real" / "reference")
    real_reference.add_argument("--openvino-model-dir", type=Path, required=True)
    real_reference.add_argument("--model-id", default=DEFAULT_REAL_MODEL_ID)
    real_reference.add_argument("--revision", required=True, help="pinned Hugging Face checkpoint revision")
    real_reference.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def prepare_output(output: Path, overwrite: bool) -> Path:
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def main() -> None:
    args = parse_args()
    output = prepare_output(args.output, args.overwrite)

    if args.command == "tiny-source":
        create_tiny_source_model(output, args.seed)
        print(f"Generated tiny Hugging Face source model: {output}")
        return

    if args.command == "tiny-reference":
        source_dir = args.source_model_dir.resolve()
        source_weights = source_dir / "model.safetensors"
        if not source_weights.is_file():
            raise FileNotFoundError(f"Expected tiny source weights at {source_weights}")
        model = load_dflash_model(source_dir)
        generate_reference(
            model,
            args.openvino_model_dir.resolve(),
            output,
            CASES["tiny"],
            args.seed,
            "tiny-dflash-pa",
            f"sha256:{sha256(source_weights)}",
        )
        print(f"Generated tiny reference fixture: {output}")
        return

    model = load_dflash_model(args.model_id, args.revision)
    generate_reference(
        model,
        args.openvino_model_dir.resolve(),
        output,
        CASES["real"],
        args.seed,
        args.model_id,
        args.revision,
    )
    print(f"Generated real reference fixture: {output}")


if __name__ == "__main__":
    main()
