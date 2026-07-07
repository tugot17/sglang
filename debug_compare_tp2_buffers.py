"""Compare dumped tp2 MoE LoRA buffers against hand-sliced expectations.

Usage:
    python debug_compare_tp2_buffers.py \
        --dump-prefix /path/to/dump \
        --adapter LiquidAI/LFM2-8B-A1B-smoltalk-LoRA \
        [--layer 2]

Expects the server to have been launched with
SGLANG_LORA_DEBUG_DUMP=/path/to/dump (and optionally
SGLANG_LORA_DEBUG_LAYER=<n>), producing /path/to/dump.moe_tp0.pt and
.moe_tp1.pt on the two ranks.

Expected buffer contents, derived directly from the raw adapter weights
(w1=gate, w3=up, w2=down; shard = moe_intermediate_size / moe_tp_size):

  A[gate_up_proj_moe][e]  rows[0:r]      = w1.lora_A          (unscaled, unsliced)
                          rows[max_r:+r] = w3.lora_A          (unscaled, unsliced)
  B[gate_up_proj_moe][e]  rows[0:shard]  = scaling * w1.lora_B[rank*shard:(rank+1)*shard]
                          rows[shard:]   = scaling * w3.lora_B[rank*shard:(rank+1)*shard]
  A[down_proj_moe][e]     rows[0:r]      = w2.lora_A[:, rank*shard:(rank+1)*shard]
  B[down_proj_moe][e]                    = scaling * w2.lora_B  (unsliced)
"""

import argparse
import glob
import json
import os
import re

import torch
from safetensors import safe_open


def find_adapter_files(adapter: str):
    if os.path.isdir(adapter):
        base = adapter
    else:
        cache = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{adapter.replace('/', '--')}"
        )
        ref = open(os.path.join(cache, "refs", "main")).read().strip()
        base = os.path.join(cache, "snapshots", ref)
    return (
        os.path.join(base, "adapter_model.safetensors"),
        os.path.join(base, "adapter_config.json"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-prefix", required=True)
    ap.add_argument("--adapter", default="LiquidAI/LFM2-8B-A1B-smoltalk-LoRA")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    st_path, cfg_path = find_adapter_files(args.adapter)
    cfg = json.load(open(cfg_path))
    scaling = cfg["lora_alpha"] / cfg["r"]
    r = cfg["r"]
    print(f"adapter: {st_path}\nscaling={scaling} r={r}")

    dumps = {}
    for path in sorted(glob.glob(args.dump_prefix + ".moe_tp*.pt")):
        d = torch.load(path, weights_only=False)
        dumps[d["moe_tp_rank"]] = d
        print(
            f"loaded {path}: moe_tp_rank={d['moe_tp_rank']}/{d['moe_tp_size']} "
            f"layer={d['layer']} scaling={d['scaling']} max_r={d['max_lora_rank']}"
        )
    assert dumps, f"no dumps found at {args.dump_prefix}.moe_tp*.pt"
    tp = next(iter(dumps.values()))["moe_tp_size"]
    layer = next(iter(dumps.values()))["layer"]
    max_r = next(iter(dumps.values()))["max_lora_rank"]

    w = {}
    with safe_open(st_path, "pt") as sf:
        pat = re.compile(
            rf"base_model\.model\.model\.layers\.{layer}\.feed_forward\."
            rf"experts\.(\d+)\.(w[123])\.lora_([AB])\.weight"
        )
        for k in sf.keys():
            m = pat.match(k)
            if m:
                w[(int(m.group(1)), m.group(2), m.group(3))] = sf.get_tensor(k).float()
    num_experts = 1 + max(e for e, _, _ in w.keys())
    inter = w[(0, "w1", "B")].shape[0]
    shard = inter // tp
    print(f"layer {layer}: {num_experts} experts, inter={inter}, shard={shard}")

    worst = {}
    for rank, d in sorted(dumps.items()):
        for e in range(num_experts):
            sl = slice(rank * shard, (rank + 1) * shard)
            exp = {
                ("A_gate_up_proj_moe", slice(0, r)): w[(e, "w1", "A")],
                ("A_gate_up_proj_moe", slice(max_r, max_r + r)): w[(e, "w3", "A")],
                ("B_gate_up_proj_moe", slice(0, shard)): scaling * w[(e, "w1", "B")][sl],
                ("B_gate_up_proj_moe", slice(shard, 2 * shard)): scaling
                * w[(e, "w3", "B")][sl],
                ("A_down_proj_moe", slice(0, r)): w[(e, "w2", "A")][:, sl],
                ("B_down_proj_moe", slice(None)): scaling * w[(e, "w2", "B")],
            }
            for (buf_name, row_slice), expected in exp.items():
                got = d[buf_name][e][row_slice]
                if buf_name.startswith("A_") or buf_name == "B_down_proj_moe":
                    got = got[..., : expected.shape[-1]]
                    got = got[: expected.shape[0]] if got.dim() == 2 else got
                if buf_name == "B_down_proj_moe":
                    got = d[buf_name][e][:, :r]
                diff = (got - expected).abs().max().item()
                key = (rank, buf_name)
                worst[key] = max(worst.get(key, 0.0), diff)

    print("\nmax abs diff vs expected (per rank / buffer):")
    ok = True
    for (rank, buf_name), diff in sorted(worst.items()):
        status = "OK" if diff <= args.tol else "MISMATCH  <-----"
        if diff > args.tol:
            ok = False
        print(f"  moe_tp_rank={rank}  {buf_name:24s} max_diff={diff:.6f}  {status}")

    if len(dumps) == 2:
        a0, a1 = dumps[0], dumps[1]
        for nm in ("A_gate_up_proj_moe",):
            d01 = (a0[nm] - a1[nm]).abs().max().item()
            print(f"\nrank0-vs-rank1 {nm} (should be identical): max_diff={d01:.6f}")

    print("\nRESULT:", "ALL BUFFERS CORRECT — bug is at runtime" if ok else "BUFFER MISMATCH — bug is at load time")


if __name__ == "__main__":
    main()
