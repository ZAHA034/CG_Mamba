"""refit_env_encoder_final.py — final-train env encoder pretrain (α.1)
================================================================================
원 train (200140-201839, 868 rows) env data 위 env encoder pretrain.
- d_model=128 (n4_d128 headline E1 winner)
- d_model=64  (n3_d64 efficiency alternative)

Held-out (201840+) 와 분리되어 leak-free. paper §V-D γ.6 의 final-train
preprocessing 정합 (E1 design-train preprocessing 와 mismatch 인정, 표준 관행).

출력:
  runs/m1_7_env_pretrain_final/env_encoder_d64.pt
  runs/m1_7_env_pretrain_final/env_encoder_d128.pt
  runs/m1_7_env_pretrain_final/diagnostics_d{D}.json
  runs/m1_7_env_pretrain_final/summary.json

비용: ~5분 GPU × 2 = ~10분
"""
from __future__ import annotations
import dataclasses
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.utils.config import CGMambaConfig
import scripts.m1_7_env_pretrain as env_pretrain_module

# Final-train 환경 (원 train fit) — design-cut 아님
FINAL_CSV       = _ROOT / "data/processed/ili_env_weekly_split.csv"
FINAL_NORM_JSON = _ROOT / "data/processed/normalization_params.json"
M1_7_ENV_OUT    = _ROOT / "runs/m1_7_env_pretrain"
FINAL_ENV_OUT   = _ROOT / "runs/m1_7_env_pretrain_final"

D_MODELS = [64, 128]                # efficiency-alt + headline only (E1 결과 기반)
EPOCHS = 100


def build_cfg(d_model: int) -> CGMambaConfig:
    return dataclasses.replace(
        CGMambaConfig(),
        d_model=d_model,
        data_csv=FINAL_CSV,
        norm_json=FINAL_NORM_JSON,
    )


def build_args() -> Namespace:
    return Namespace(smoke=False, epochs=EPOCHS, log_every=10)


def main():
    FINAL_ENV_OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== env encoder final pretrain (α.1, 원 train 200140-201839) ===")
    print(f"  CSV  = {FINAL_CSV.name}")
    print(f"  norm = {FINAL_NORM_JSON.name}")
    print(f"  d_model grid = {D_MODELS}  epochs = {EPOCHS}")

    results = {}
    for d_model in D_MODELS:
        print(f"\n{'='*70}\n[env pretrain final] d_model={d_model}\n{'='*70}", flush=True)
        cfg = build_cfg(d_model)
        args = build_args()
        diagnostics = env_pretrain_module.pretrain_env(cfg, args)
        src_ckpt = M1_7_ENV_OUT / "env_encoder.pt"
        src_diag = M1_7_ENV_OUT / "diagnostics.json"
        dst_ckpt = FINAL_ENV_OUT / f"env_encoder_d{d_model}.pt"
        dst_diag = FINAL_ENV_OUT / f"diagnostics_d{d_model}.json"
        assert src_ckpt.exists(), f"missing pretrain output: {src_ckpt}"
        shutil.move(str(src_ckpt), str(dst_ckpt))
        if src_diag.exists():
            shutil.move(str(src_diag), str(dst_diag))
        vl = diagnostics.get("val_mse_final", diagnostics.get("final_val_mse")) if isinstance(diagnostics, dict) else None
        results[d_model] = dict(val_mse_final=vl, ckpt=str(dst_ckpt.relative_to(_ROOT)))
        print(f"  d_model={d_model}  val_mse={vl}", flush=True)

    print(f"\n=== final env pretrain SUMMARY ===")
    for d_model in D_MODELS:
        r = results[d_model]
        print(f"  d_model={d_model:3d}  ckpt={r['ckpt']}  val_mse={r['val_mse_final']}")

    with open(FINAL_ENV_OUT / "summary.json", "w") as f:
        json.dump(dict(d_models=D_MODELS, epochs=EPOCHS,
                         data_csv=str(FINAL_CSV.relative_to(_ROOT)),
                         norm_json=str(FINAL_NORM_JSON.relative_to(_ROOT)),
                         results=results), f, indent=2, default=str)
    print(f"\n  saved: {FINAL_ENV_OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
