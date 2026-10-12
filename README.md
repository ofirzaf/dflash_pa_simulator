# GPU PagedAttention: MIXED stage with `token_type_ids`

Reproducer for [CVS-196087](https://jira.devtools.intel.com/browse/CVS-196087). `pa_mixed_token_type.py` runs
one `PagedAttentionExtension` op on the GPU and compares it with a NumPy reference. It needs no model files.

The op and its compiled form match a layer of the Qwen3.5-9B INT4 DFlash draft run through GenAI with default
properties: 32 Q heads, 8 KV heads, head size 128, B = 16, a 4096 sliding window (5 of 6 layers), and a u4 KV
cache. See `--help` to change them.

## Case

Each sequence in a packed call has these Q/K/V rows:

```text
rows:            | D context rows | B block rows |
token_type_ids:  | 0 ... 0        | 1 ... 1      |
```

- Type 0 is causal (and within the sliding window, if any).
- Type 1 also sees its whole contiguous type-1 run (bidirectional block).
- Only the D rows are committed: the next step starts at `past_len + D` and overwrites the block rows in the cache.

Every decode step has `past_len > 0` and more than one row, so it always runs in the MIXED stage.

## Run

```bash
pip install -r requirements.txt   # or use your own OpenVINO build
python pa_mixed_token_type.py check
python pa_mixed_token_type.py bench
```

`check` runs a multi-step, multi-sequence schedule for KV caches u4, i8 and f16 and windows 0, 4096 and 32, and
exits with 1 on failure. On master (`2026.5.0-23394`, Arc B390) every MIXED step fails on the block rows;
PREFILL and context rows pass:

```text
== kv_cache=u4 sliding_window=0 atol=1.0 rtol=0.0 ==
step 0 prefill, one sequence    stage=PREFILL past=[0]  rows=[53] max_abs context=0.0011 block=0.0007 PASS
step 1 decode, one sequence     stage=MIXED   past=[37] rows=[19] max_abs context=0.1301 block=3.0915 FAIL
```

`bench` times one MIXED step with the u4 cache over `--batch`, `--past` and `--context-rows` (median wall time
of `infer()`, including host overhead). The `none` rows use an op without `token_type_ids`, i.e. plain causal
MIXED.

## Notes

- Release wheels don't show whether micro SDPA or `paged_attention_opt.cl` ran; use a build with
  `ENABLE_DEBUG_CAPS=ON` and `OV_VERBOSE=4`.
- `paged_attention_token_type_micro_sdpa_mixed_test` in `paged_attention_token_type_gpu_test.cpp` picks
  `past_len` so all new rows are type 0, so it misses this case.
