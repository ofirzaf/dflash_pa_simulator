# DFlash PagedAttention GPU Simulator

This is a standalone GPU kernel reproducer. It does not include, link, build,
or call OpenVINO GenAI.

It takes a stateful DFlash draft IR, applies OpenVINO's native
`SDPAToPagedAttention` pass, allocates only GPU FP16 paged KV tensors, and
runs a fixed two-step, one-sequence DFlash case:

```text
Q = B
current K/V = D + B
full cache = past + D + B
committed cache after validation = past + D
```

The seed-plus-proposal `B` tail is excluded from the next block table. A
partially committed cache block remains mapped; a fully temporary block becomes
available for reuse on the second step.

When the native pass emits it, the simulator also binds
`score_aggregation_window={1}` alongside `past_lens`, `subsequence_begins`,
`block_indices`, `block_indices_begins`, and `max_context_len`.

## Requirements

- A GPU-capable OpenVINO build that exposes the `GPU` device and includes
  `openvino/pass/sdpa_to_paged_attention.hpp`.
- PyTorch with `torch.xpu.is_available()` for fixture generation.
- `optimum-cli` from the [`dflash-pa`](https://github.com/ofirzaf/optimum-intel/tree/dflash-pa)
  branch of Optimum Intel.
- Python 3.12, a C++17 compiler, and Ninja.

The simulator is intentionally GPU-only. It rejects non-GPU devices and
requires FP16 GPU KV cache inputs with a 16-token block dimension.

## Environment setup with uv

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --prerelease=allow \
  -r dflash_pa_simulator/requirements.txt
```

Use the branch that exports the full-cache DFlash IR:

```bash
git clone --branch dflash-pa https://github.com/ofirzaf/optimum-intel.git optimum-intel
uv pip install --index-strategy unsafe-best-match --prerelease=allow -e ./optimum-intel
```

If this workspace already contains `optimum-intel`, switch it to `dflash-pa`
and reinstall it in the active uv environment instead.

`unsafe-best-match` is required because PyPI exposes a CPU-only `torch`
package while the requirements file selects the XPU wheel from Intel's index.

## Build

```bash
uv run cmake -S dflash_pa_simulator -B dflash_pa_simulator/build -G Ninja \
  -DOpenVINO_DIR="$(uv run python -c 'from openvino.utils import get_cmake_path; print(get_cmake_path())')"
uv run cmake --build dflash_pa_simulator/build --parallel
```

For kernel development, point `OpenVINO_DIR` at the kernel team's OpenVINO
build rather than the Python package.

## Generate and export the tiny source model

The generator first creates a standard Hugging Face/Transformers DFlash
checkpoint with `config.json` and `model.safetensors`. It does not construct
an OpenVINO model or call Optimum's Python export API.

All generated source models, OpenVINO IRs, and reference tensors below live
under `dflash_pa_simulator/target/`, which is ignored by Git.

```bash
uv run python dflash_pa_simulator/generate_fixture.py tiny-source \
  --output dflash_pa_simulator/target/tiny/source --overwrite
```

Export that source through the exact CLI path used for an external draft:

```bash
uv run optimum-cli export openvino \
  --model dflash_pa_simulator/target/tiny/source \
  --trust-remote-code \
  --weight-format fp16 \
  --task text-generation-with-past \
  dflash_pa_simulator/target/tiny/model
```

The tiny model has two FP16 DFlash layers: one full-attention layer and one
sliding-window layer. It uses `D0=15`, `D1=3`, and `B=4`, crossing both the
small window and the 16-token GPU cache-block boundary.

After CLI export, generate the XPU reference fixture from the same source
weights:

```bash
uv run python dflash_pa_simulator/generate_fixture.py tiny-reference \
  --source-model-dir dflash_pa_simulator/target/tiny/source \
  --openvino-model-dir dflash_pa_simulator/target/tiny/model \
  --output dflash_pa_simulator/target/tiny/reference --overwrite
```

## Generate the real-model reference fixture

First export `z-lab/Qwen3.5-9B-DFlash` using the same `dflash-pa` CLI:

```bash
uv run optimum-cli export openvino \
  --model z-lab/Qwen3.5-9B-DFlash \
  --trust-remote-code \
  --weight-format fp16 \
  --task text-generation-with-past \
  dflash_pa_simulator/target/real/model
```

Then generate its XPU acceptance fixture:

```bash
uv run python dflash_pa_simulator/generate_fixture.py real-reference \
  --revision 5fc3b3d474760f18c516db87d84c37edbfd3ede6 \
  --openvino-model-dir dflash_pa_simulator/target/real/model \
  --output dflash_pa_simulator/target/real/reference --overwrite
```

The fixture generator runs the PyTorch reference on XPU in FP16. It stores
FP16 raw tensors, model SHA-256 checksums, expected full caches, and expected
committed prefixes. The raw IR's public activation ports are FP32, so the C++
tool widens fixture inputs at that boundary and compares the FP32 output to the
widened FP16 PyTorch output.

## Run

```bash
uv run dflash_pa_simulator/build/dflash_pa_simulator \
  --model dflash_pa_simulator/target/tiny/model \
  --fixture dflash_pa_simulator/target/tiny/reference \
  --device GPU \
  --dump-pa-ir
```

For the real acceptance case:

```bash
uv run dflash_pa_simulator/build/dflash_pa_simulator \
  --model dflash_pa_simulator/target/real/model \
  --fixture dflash_pa_simulator/target/real/reference \
  --device GPU
```

`--atol` and `--rtol` default to `0.1` for FP16 cross-backend comparison.

## Diagnostics

Failures identify one of these stages:

1. fixture/model checksum or FP16-weight validation;
2. native PA transformation;
3. transformed-model ABI validation;
4. GPU compilation;
5. GPU inference;
6. numerical comparison.

For this fixed batch-1, no-padding fixture, the simulator replaces
`beam_idx` with `{0}` and `attention_mask` with an all-valid constant before
the native PA pass. Those stateful-SDPA controls no longer block conversion.

With the baseline OpenVINO runtime used to create this project, the pass then
completes but matches none of the DFlash SDPAs. The remaining raw graph has
asymmetric rank-5 Q/K/V layout, while the standard matcher expects its normal
4-D/flattened layout. This is now reported as a matcher/PA-ABI gap for
`Q=B`, `K/V=D+B`, rather than as a `beam_idx` or `attention_mask` failure.
`--dump-pa-ir` prints the raw SDPA ports now and the PA port contract once the
matcher converts the graph.
