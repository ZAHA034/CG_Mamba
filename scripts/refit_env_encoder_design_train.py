"""refit_env_encoder_design_train.py — γ.4채널 (env pretrain) 누설 차단
================================================================================
원 env_encoder.pt: m1_7_env_pretrain 가 *full train (200140-201839)* 위 fit
  → design-val (201540-201839) 구간 env 통계 누설.

조치: design-train (split=='train' AND ep ≤ 201539, 712 rows) 만으로 env encoder
재pretrain. d_model 별 3개 (32/64/128) — EnvModule.encoder 의 두 번째 Linear
가 d_model 의존 (Linear(H=32, D=d_model)).

출력:
  runs/m1_7_env_pretrain_design/env_encoder_d32.pt
  runs/m1_7_env_pretrain_design/env_encoder_d64.pt
  runs/m1_7_env_pretrain_design/env_encoder_d128.pt
  runs/m1_7_env_pretrain_design/diagnostics_d{D}.json (per d_model)

비용: ~5-10분 GPU × 3 = ~30분
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

DESIGN_CSV       = _ROOT / "data/processed/ili_env_weekly_split_design.csv"
DESIGN_NORM_JSON = _ROOT / "data/processed/normalization_params_design_train.json"
M1_7_ENV_OUT     = _ROOT / "runs/m1_7_env_pretrain"               # m1_7_env_pretrain.py hardcoded
DESIGN_ENV_OUT   = _ROOT / "runs/m1_7_env_pretrain_design"

D_MODELS = [32, 64, 128]
EPOCHS = 100                  # default in m1_7_env_pretrain (env 가 작은 module, 충분히 수렴)


def build_pretrain_cfg(d_model: int) -> CGMambaConfig:
    base = CGMambaConfig()
    return dataclasses.replace(
        base,
        d_model=d_model,
        data_csv=DESIGN_CSV,
        norm_json=DESIGN_NORM_JSON,
    )


def build_pretrain_args(d_model: int) -> Namespace:
    # m1_7_env_pretrain.main 의 argparse 모방
    return Namespace(
        smoke=False,
        epochs=EPOCHS,
        log_every=10,
    )


def main():
    DESIGN_ENV_OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== env encoder design-train 재pretrain (γ.5 채널 4) ===")
    print(f"  CSV  = {DESIGN_CSV.name}")
    print(f"  norm = {DESIGN_NORM_JSON.name}")
    print(f"  d_model grid = {D_MODELS}  epochs = {EPOCHS}")
    print(f"  output dir = {DESIGN_ENV_OUT}")

    results = {}
    for d_model in D_MODELS:
        print(f"\n{'='*70}")
        print(f"[env pretrain] d_model={d_model}  epochs={EPOCHS}")
        print(f"{'='*70}", flush=True)

        cfg = build_pretrain_cfg(d_model)
        args = build_pretrain_args(d_model)

        # In-process call
        diagnostics = env_pretrain_module.pretrain_env(cfg, args)

        # m1_7_env_pretrain 가 runs/m1_7_env_pretrain/env_encoder.pt 에 hardcoded 저장.
        # → design-cut 결과를 design-cut dir 로 mv (d_model suffix 붙임)
        src_ckpt = M1_7_ENV_OUT / "env_encoder.pt"
        src_diag = M1_7_ENV_OUT / "diagnostics.json"
        dst_ckpt = DESIGN_ENV_OUT / f"env_encoder_d{d_model}.pt"
        dst_diag = DESIGN_ENV_OUT / f"diagnostics_d{d_model}.json"

        assert src_ckpt.exists(), f"missing pretrain output: {src_ckpt}"
        shutil.move(str(src_ckpt), str(dst_ckpt))
        if src_diag.exists():
            shutil.move(str(src_diag), str(dst_diag))

        # final metrics 요약
        if isinstance(diagnostics, dict):
            tr_final = diagnostics.get("train_mse_final", diagnostics.get("final_train_mse"))
            vl_final = diagnostics.get("val_mse_final", diagnostics.get("final_val_mse"))
            print(f"  d_model={d_model}  final train_mse={tr_final}  val_mse={vl_final}")
            results[d_model] = dict(train_mse_final=tr_final, val_mse_final=vl_final,
                                      ckpt=str(dst_ckpt.relative_to(_ROOT)))
        else:
            results[d_model] = dict(ckpt=str(dst_ckpt.relative_to(_ROOT)))

    # Aggregate summary
    print(f"\n{'='*70}")
    print(f"=== env pretrain SUMMARY (design-train only, 4채널 누설 차단) ===")
    print(f"{'='*70}")
    for d_model in D_MODELS:
        r = results[d_model]
        print(f"  d_model={d_model:3d}  ckpt={r['ckpt']}  "
              f"train_mse={r.get('train_mse_final', 'NA')}  val_mse={r.get('val_mse_final', 'NA')}")

    summary_path = DESIGN_ENV_OUT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(dict(
            d_models=D_MODELS, epochs=EPOCHS,
            data_csv=str(DESIGN_CSV.relative_to(_ROOT)),
            norm_json=str(DESIGN_NORM_JSON.relative_to(_ROOT)),
            results=results,
        ), f, indent=2, default=str)
    print(f"\n  saved: {summary_path}")


if __name__ == "__main__":
    main()
