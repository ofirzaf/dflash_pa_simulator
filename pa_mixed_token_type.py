#!/usr/bin/env python3
"""GPU PagedAttentionExtension in the MIXED stage with token_type_ids, shaped like a DFlash draft step.

Each sequence in a packed call holds D context rows (token type 0, causal) followed by B block rows
(token type 1, bidirectional within the block). Only the D context rows are committed; the next step starts
at past + D and overwrites the block rows in the cache.

The op and its compiled form match the Qwen3.5-9B INT4 DFlash draft run through GenAI with default properties,
including its u4 KV cache.

  check  compares the GPU output with a NumPy reference over a multi-step, multi-sequence schedule
  bench  times one MIXED step at realistic context lengths and batch sizes
"""

import argparse
import itertools
import statistics
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops
from openvino.op import _PagedAttentionExtension

PREFILL, MIXED, GENERATE = "PREFILL", "MIXED", "GENERATE"
KV_CACHE_PRECISIONS = ("u4", "i8", "f16")
# correct kernels stay below these (worst seen: u4 0.41, i8 0.018, f16 0.007); a block masking error is above 2
ATOL = {"u4": 1.0, "i8": 5e-2, "f16": 1e-2}


def stage(past_lens, row_counts):
    if all(rows == 1 for rows in row_counts):
        return GENERATE
    return MIXED if any(past_lens) else PREFILL


def build_paged_attention(core, args, window, kv_precision, with_token_types=True):
    """One PagedAttentionExtension as SDPAToPagedAttention emits it for the draft, compiled with default properties.

    KV_CACHE_PRECISION is set explicitly: the plugin picks u4 for the INT4 draft, but this op has no weights,
    so its own default would be i8.
    """

    def parameter(name, shape, dtype=ov.Type.i32):
        return ops.parameter(shape, dtype, name=name).output(0)

    def empty(dtype=ov.Type.i32, shape=(0,)):
        return ops.constant(np.empty(shape, dtype=dtype.to_dtype())).output(0)

    def scalar(value, dtype=ov.Type.i32):
        return ops.constant(value, dtype).output(0)

    q_width, kv_width = args.q_heads * args.head_size, args.kv_heads * args.head_size
    parameters = []
    if with_token_types:
        # the draft feeds i64 [rows, 1] and the graph converts it to i32
        parameters.append(ops.parameter([-1, 1], ov.Type.i64, name="token_type_ids"))
        token_type_ids = ops.convert(parameters[-1], ov.Type.i32).output(0)
    else:
        token_type_ids = empty()
    inputs = [
        parameter("query", [-1, q_width], ov.Type.f16),  # 0
        parameter("key", [-1, kv_width], ov.Type.f16),  # 1
        parameter("value", [-1, kv_width], ov.Type.f16),  # 2
        parameter("key_cache.0", [-1, -1, -1, -1], ov.Type.dynamic),  # 3
        parameter("value_cache.0", [-1, -1, -1, -1], ov.Type.dynamic),  # 4
        parameter("past_lens", [-1]),  # 5
        parameter("subsequence_begins", [-1]),  # 6
        parameter("block_indices", [-1]),  # 7
        parameter("block_indices_begins", [-1]),  # 8
        scalar(1 / np.sqrt(args.head_size), ov.Type.f32),  # 9 scale
        scalar(window),  # 10 sliding_window
        empty(ov.Type.f32),  # 11 alibi_slopes
        parameter("max_context_len", []),  # 12
        empty(),  # 13 score_aggregation_window
        empty(),  # 14 rotated_block_indices
        empty(),  # 15 rotation_deltas
        empty(ov.Type.f32),  # 16 rotation_trig_lut
        empty(ov.Type.f32),  # 17 xattention_threshold
        scalar(0),  # 18 xattention_block_size
        scalar(0),  # 19 xattention_stride
        empty(ov.Type.f32, (0, 0, 0, 0)),  # 20 sinks
        scalar(0),  # 21 adaptive_rkv_start_size
        empty(),  # 22 adaptive_rkv_evictable_sizes
        empty(),  # 23 adaptive_rkv_diversity_block_set_indices
        empty(),  # 24 adaptive_rkv_diversity_block_set_indices_begins
        token_type_ids,  # 25
        empty(ov.Type.u8),  # 26 qq_bias
        empty(),  # 27 qq_bias_begins
    ]
    paged_attention = _PagedAttentionExtension(inputs)
    for key, value in {
        "num_k_heads": args.kv_heads,
        "num_v_heads": args.kv_heads,
        "k_head_size": args.head_size,
        "v_head_size": args.head_size,
    }.items():
        paged_attention.get_rt_info()[key] = value
    parameters += [port.get_node() for port in inputs if port.get_node().get_type_name() == "Parameter"]
    model = ov.Model([paged_attention.output(0)], parameters, f"pa_window_{window}")
    return core.compile_model(model, args.device, {"KV_CACHE_PRECISION": kv_precision})


class PagedCache:
    """One device KV cache per layer, with pages handed to sequences in shuffled order."""

    def __init__(self, core, device, compiled, sequence_lengths, rng):
        """sequence_lengths: the most tokens each sequence will hold at once."""
        self.request = compiled.create_infer_request()
        caches = {port.any_name: port for port in compiled.inputs if port.any_name.startswith(("key_cache.", "value_cache."))}
        self.block_size = caches["value_cache.0"].partial_shape[2].get_length()  # [blocks, kv_heads, block_size, head_size]
        num_blocks = sum(-(-length // self.block_size) for length in sequence_lengths)
        context = core.get_default_context(device)
        for name, port in caches.items():
            dims = [port.partial_shape[i].get_length() for i in range(1, 4)]
            # host tensors are not written back, so the cache must live on the device
            self.request.set_tensor(name, context.create_tensor(port.element_type, ov.Shape([num_blocks, *dims]), {}))
        self.free = list(rng.permutation(num_blocks))
        self.tables = {}

    def share(self, compiled):
        """An infer request of another compiled op that reads and writes the same cache tensors."""
        request = compiled.create_infer_request()
        for port in compiled.inputs:
            if port.any_name.startswith(("key_cache.", "value_cache.")):
                request.set_tensor(port.any_name, self.request.get_tensor(port.any_name))
        return request

    def pages(self, seq_id, length):
        table = self.tables.setdefault(seq_id, [])
        while len(table) * self.block_size < length:
            if not self.free:
                raise RuntimeError("paged KV cache is full")
            table.append(self.free.pop())
        return table[: -(-length // self.block_size)]

    def feed(self, sequences):
        """sequences: list of (seq_id, past_len, query, key, value, token_types) with f16 2D activations."""
        begins, block_indices, block_begins = [0], [], [0]
        for seq_id, past, query, _, _, _ in sequences:
            begins.append(begins[-1] + query.shape[0])
            block_indices += self.pages(seq_id, past + query.shape[0])
            block_begins.append(len(block_indices))
        return {
            "query": np.concatenate([s[2] for s in sequences]),
            "key": np.concatenate([s[3] for s in sequences]),
            "value": np.concatenate([s[4] for s in sequences]),
            "past_lens": np.array([s[1] for s in sequences], np.int32),
            "subsequence_begins": np.array(begins, np.int32),
            "block_indices": np.array(block_indices, np.int32),
            "block_indices_begins": np.array(block_begins, np.int32),
            "max_context_len": np.array(max(s[1] + s[2].shape[0] for s in sequences), np.int32),
            "token_type_ids": np.concatenate([s[5] for s in sequences]).astype(np.int64)[:, None],
        }


def make_rows(rng, args, context_rows, block_rows):
    """Random f16 Q/K/V for D context rows and B block rows.

    Block queries and keys share a per-head direction that adds --block-affinity to their attention scores,
    so block rows put most of their weight on the block and a block masking error is visible at any context length.
    """
    rows = context_rows + block_rows
    q = rng.standard_normal((rows, args.q_heads, args.head_size), np.float32)
    k = rng.standard_normal((rows, args.kv_heads, args.head_size), np.float32)
    v = rng.standard_normal((rows, args.kv_heads, args.head_size), np.float32)
    if args.block_affinity > 0:
        direction = rng.standard_normal((args.kv_heads, args.head_size), np.float32)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        direction *= np.sqrt(args.block_affinity * np.sqrt(args.head_size))
        group = args.q_heads // args.kv_heads
        q[context_rows:] += np.repeat(direction, group, axis=0)
        k[context_rows:] += direction
    types = np.array([0] * context_rows + [1] * block_rows, np.int32)
    return q.astype(np.float16), k.astype(np.float16), v.astype(np.float16), types


def reference_attention(q, k, v, past, types, window, kv_heads, chunk=256):
    """Causal attention over [past; current] rows. A token of type 1 also sees its whole contiguous run of
    type-1 tokens among the current rows, regardless of the sliding window.

    q: [rows, q_heads, head_size]; k, v: [past + rows, kv_heads, head_size]; returns [rows, q_heads * head_size].
    """
    rows, q_heads, head_size = q.shape
    group = q_heads // kv_heads
    total = past + rows
    run_begin, run_end = np.arange(rows), np.arange(rows) + 1
    for i in range(1, rows):
        if types[i] and types[i - 1]:
            run_begin[i] = run_begin[i - 1]
    for i in range(rows - 2, -1, -1):
        if types[i] and types[i + 1]:
            run_end[i] = run_end[i + 1]
    keys = k.transpose(1, 2, 0)  # [kv_heads, head_size, total]
    values = v.transpose(1, 0, 2)  # [kv_heads, total, head_size]
    key_pos = np.arange(total)
    output = np.empty((rows, q_heads * head_size), np.float32)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        pos = past + np.arange(start, stop)
        visible = key_pos[None] <= pos[:, None]
        if window:
            visible &= key_pos[None] > pos[:, None] - window
        for i in range(start, stop):
            if types[i]:
                visible[i - start, past + run_begin[i] : past + run_end[i]] = True
        query = q[start:stop].reshape(stop - start, kv_heads, group, head_size).transpose(1, 0, 2, 3)
        scores = (query.reshape(kv_heads, -1, head_size) @ keys).reshape(kv_heads, stop - start, group, total)
        scores /= np.sqrt(head_size)
        scores = np.where(visible[None, :, None, :], scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out = (probs.reshape(kv_heads, -1, total) @ values).reshape(kv_heads, stop - start, group, head_size)
        output[start:stop] = out.transpose(1, 0, 2, 3).reshape(stop - start, -1)
    return output


def check_schedule(args):
    """(description, [(seq_id, context_rows), ...]) per step; sequence 2 crosses the --sliding-window edge."""
    block = args.block_len
    long_prefill = max(args.sliding_window, 4 * block) - 10
    return [
        ("prefill, one sequence", [(0, 37)]),
        ("decode, one sequence", [(0, 3)]),
        ("decode, all drafts rejected", [(0, 1)]),
        ("decode, all drafts accepted", [(0, block)]),
        ("decode + prefill packed", [(0, 5), (1, 50)]),
        ("two decodes packed", [(0, 2), (1, 9)]),
        ("decode + long prefill packed", [(0, 4), (2, long_prefill)]),
        ("decode across the window edge", [(2, 6), (1, 1)]),
        ("decode past the window", [(2, 11), (0, 7)]),
    ]


def run_check(core, args):
    schedule = check_schedule(args)
    windows = list(dict.fromkeys([0, args.sliding_window, args.extra_window]))
    windows = [w for i, w in enumerate(windows) if w or i == 0]
    lengths, peak = {}, {}
    for _, step in schedule:
        for seq_id, context_rows in step:
            peak[seq_id] = max(peak.get(seq_id, 0), lengths.get(seq_id, 0) + context_rows + args.block_len)
            lengths[seq_id] = lengths.get(seq_id, 0) + context_rows
    failures = 0
    for kv_precision, window in itertools.product(args.kv_cache_precision, windows):
        atol = ATOL[kv_precision] if args.atol is None else args.atol
        compiled = build_paged_attention(core, args, window, kv_precision)
        rng = np.random.default_rng(args.seed)
        cache = PagedCache(core, args.device, compiled, peak.values(), rng)
        history = {}
        print(f"\n== kv_cache={kv_precision} sliding_window={window} atol={atol} rtol={args.rtol} ==")
        for index, (description, step) in enumerate(schedule):
            sequences = []
            for seq_id, context_rows in step:
                past = 0 if seq_id not in history else history[seq_id][0].shape[0]
                sequences.append((seq_id, past, *make_rows(rng, args, context_rows, args.block_len)))
            cache.request.infer(cache.feed([(s, p, q.reshape(q.shape[0], -1), k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1), t) for s, p, q, k, v, t in sequences]))
            actual = cache.request.get_output_tensor().data.astype(np.float32)
            offset, errors = 0, {0: 0.0, 1: 0.0}
            step_failed = False
            for (seq_id, context_rows), (_, past, q, k, v, types) in zip(step, sequences):
                rows = q.shape[0]
                old_k, old_v = history.get(seq_id, (k[:0].astype(np.float32), v[:0].astype(np.float32)))
                full_k = np.concatenate([old_k, k.astype(np.float32)])
                full_v = np.concatenate([old_v, v.astype(np.float32)])
                expected = reference_attention(q.astype(np.float32), full_k, full_v, past, types, window, args.kv_heads)
                got = actual[offset : offset + rows]
                offset += rows
                for token_type in (0, 1):
                    selected = types == token_type
                    errors[token_type] = max(errors[token_type], float(np.abs(got[selected] - expected[selected]).max()))
                if not np.allclose(got, expected, atol=atol, rtol=args.rtol):
                    step_failed = True
                history[seq_id] = (full_k[: past + context_rows], full_v[: past + context_rows])
            failures += step_failed
            pasts = [s[1] for s in sequences]
            rows = [s[2].shape[0] for s in sequences]
            print(
                f"step {index} {description:32s} stage={stage(pasts, rows):7s} past={pasts} rows={rows} "
                f"max_abs context={errors[0]:.4f} block={errors[1]:.4f} {'FAIL' if step_failed else 'PASS'}",
                flush=True,
            )
    print(f"\n{'FAIL' if failures else 'PASS'}: {failures} failing step(s)")
    return 1 if failures else 0


def run_bench(core, args):
    block = args.block_len
    max_rows = max(args.context_rows) + block
    max_past, max_batch = max(args.past), max(args.batch)
    print("\nMedian wall time of one synchronous infer() of a single PA op; includes host overhead.")
    print("'none' is the same step on an op without the token_type_ids input (plain causal MIXED).")
    print(f"\n{'kv':>3} {'window':>6} {'batch':>5} {'past':>6} {'D':>3} {'B':>3} {'token_type_ids':>14} {'ms':>8}")
    windows = [0, args.sliding_window] if args.sliding_window else [0]
    for kv_precision, window in itertools.product(args.kv_cache_precision, windows):
        rng = np.random.default_rng(args.seed)
        compiled = build_paged_attention(core, args, window, kv_precision)
        cache = PagedCache(core, args.device, compiled, [max_past + max_rows] * max_batch, rng)
        plain = cache.share(build_paged_attention(core, args, window, kv_precision, with_token_types=False))
        # fill every sequence's history once; a shorter past reuses the first pages of the same sequence
        q, k, v, _ = make_rows(rng, args, max_past, 0)
        for seq_id in range(max_batch):
            cache.request.infer(cache.feed([(seq_id, 0, q.reshape(max_past, -1), k.reshape(max_past, -1), v.reshape(max_past, -1), np.zeros(max_past, np.int32))]))
        for batch, past, context_rows in itertools.product(args.batch, args.past, args.context_rows):
            q, k, v, types = make_rows(rng, args, context_rows, block)
            q, k, v = (x.reshape(x.shape[0], -1) for x in (q, k, v))
            feed = cache.feed([(seq_id, past, q, k, v, types) for seq_id in range(batch)])
            for label, request in (("[0]*D+[1]*B", cache.request), ("none", plain)):
                names = {port.any_name for port in request.model_inputs}
                for name, value in feed.items():
                    if name in names:
                        request.set_tensor(name, ov.Tensor(value))
                wall = []
                for iteration in range(args.warmup + args.iters):
                    start = time.perf_counter()
                    request.infer()
                    if iteration >= args.warmup:
                        wall.append((time.perf_counter() - start) * 1e3)
                print(
                    f"{kv_precision:>3} {window:>6} {batch:>5} {past:>6} {context_rows:>3} {block:>3} {label:>14} "
                    f"{statistics.median(wall):>8.3f}",
                    flush=True,
                )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=["check", "bench"], default="check")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--sliding-window", type=int, default=4096, help="window of the draft's sliding layers (0: none)")
    parser.add_argument("--block-len", type=int, default=16, help="block rows B per sequence (DFlash block size)")
    parser.add_argument("--block-affinity", type=float, default=4.0, help="score added between block rows (0: off)")
    parser.add_argument(
        "--kv-cache-precision",
        nargs="+",
        choices=KV_CACHE_PRECISIONS,
        help="u4 is what the INT4 draft gets with default properties (default: check all, bench u4)",
    )
    parser.add_argument("--seed", type=int, default=0)
    check = parser.add_argument_group("check")
    check.add_argument("--extra-window", type=int, default=32, help="also check a small window (0: skip)")
    check.add_argument("--atol", type=float, help=f"default per KV cache precision: {ATOL}")
    check.add_argument("--rtol", type=float, default=0.0)
    bench = parser.add_argument_group("bench")
    bench.add_argument("--batch", type=int, nargs="+", default=[1, 4], help="sequences per call")
    bench.add_argument("--past", type=int, nargs="+", default=[1024, 4096, 16384], help="committed length per sequence")
    bench.add_argument("--context-rows", type=int, nargs="+", default=[4], help="context rows D per sequence")
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be a multiple of --kv-heads")
    if args.kv_cache_precision is None:
        args.kv_cache_precision = list(KV_CACHE_PRECISIONS) if args.mode == "check" else ["u4"]

    core = ov.Core()
    print(f"OpenVINO {ov.get_version()}")
    print(f"Device {args.device}: {core.get_property(args.device, 'FULL_DEVICE_NAME')}")
    print(f"q_heads={args.q_heads} kv_heads={args.kv_heads} head_size={args.head_size} block_len={args.block_len} "
          f"sliding_window={args.sliding_window} block_affinity={args.block_affinity}")
    return run_check(core, args) if args.mode == "check" else run_bench(core, args)


if __name__ == "__main__":
    sys.exit(main())
