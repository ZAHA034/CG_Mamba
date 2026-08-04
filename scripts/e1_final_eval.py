"""scripts/e1_final_eval.py — (α.3) E1 final-train Multi-range Eval
================================================================================
α.2 가 저장한 10 ckpts (n2_d128 × 5 + n3_d64 × 5) 을 3 범위 평가:

  - **n2_d128**: 사전등록 누설-free HPO 의 winner (γ.3 val_total ascending) = HEADLINE
  - **n3_d64**: paper-config robustness check (= paper m1_9 / ablation_retrain 의 config)

Reversal 인지 (2026-06-18 6축 audit 보존):
  design-split HPO winner = n2_d128 (val_total=0.3030±0.0136)
  full-retrain val_total: n2_d128=0.4130 > n3_d64=0.3906 (full-train 에서 reversal)
  → val/test rank divergence (m1_9 의 per-base top-K=2 diversification 도입 이유)
  → reversal 은 finding 으로 disclose. headline crown 은 사전등록 그대로 n2_d128.

D.1 fix (per-seed HMM, 2026-06-18 train audit): paper ablation_retrain mirror.
D.4 fix (per-seed HMM in eval load_final_model, 2026-06-18 eval audit):
  학습은 per-seed HMM 인데 eval APMD 분해 입력 μ_k/s²_k 가 seed42 HMM stats 였음
  → self-inconsistent. fix: load_fitted_hmm 도 per-seed dispatch.

1) **held-out (W40-2018 ~ W10-2020, national)** — FluSight 2018-19 포함
   - raw 만 보고 (cal-set lock 사전등록 γ.7 #6)
   - 순위는 점예측 (mu) 기반이라 head=σ² 무관

2) **test_strict national (W40-2022 ~ W35-2025)**
   - **raw + E3-calibrated dual report** (cal-set: held-out 2018-2020)
   - paper Table I 헤드라인 (n2_d128 사전등록 winner 가 진짜 headline)

3) **test_strict regional (hhs1-10, zero-shot transfer)**
   - **raw + E3-calibrated dual report**
   - paper Table IV transfer 헤드라인

4) **PC2-a 재확인** (n2_d128 winner 만, regional test_strict, phase-anchored vs s_h)
   - raw HMM 분산 only (γ.5 narrative lock — selection-level mechanism test)
   - region-cluster bootstrap (src/eval/bootstrap util)
   - buggy 시 0.149 → corrected E1 fix 후 회복 여부 결정타

Output: runs/e1_final/e1_final_eval.json + per-(config, seed, range) parquet
"""
from __future__ import annotations
import dataclasses
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as sp_norm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.models.cg_forecaster import CGForecaster
from src.models.heteroscedastic_head import (
    HeteroHead, fit_hetero_head, apply_hetero_head, eval_cov95_wis, FLUSIGHT_23,
)
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import load_dataset_csv, load_norm_params, MultiHorizonDataset
from src.eval.bootstrap import cluster_bootstrap_mean, cluster_bootstrap_delta
from torch.utils.data import DataLoader

import regime_shift_drivers as rsd                 # _build_region_df 등 재활용

# === LOCKED paths (γ.6 final preprocessing) ===
FINAL_CSV       = _ROOT / "data/processed/ili_env_weekly_split.csv"
FINAL_NORM_JSON = _ROOT / "data/processed/normalization_params.json"
# D.4 fix (2026-06-18 eval audit): per-seed HMM template (학습 시 ckpt 와 self-consistent)
FINAL_HMM_TPL   = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"
OUT_DIR         = _ROOT / "runs/e1_final"

# E1 4차 HPO winner 확정 (2026-06-18): n2_d128 사전등록 누설-free headline / n3_d64 robustness check
CONFIGS = [
    ("n2_d128", 2, 128),     # 사전등록 누설-free HPO winner = HEADLINE (val/test rank reversal disclose)
    ("n3_d64",  3,  64),     # paper-config robustness check (clean ≈ leaky 카드: val 예고편, test 확정 대기)
]
HEADLINE_CONFIG_ID = "n2_d128"   # PC2-a recheck 가 적용될 winner
SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]
REGIONS_TRANSFER = [f"hhs{i}" for i in range(1, 11)]
TEST_STRICT_START_EPIWEEK = 202240        # γ.1
FLUSIGHT_2018_19_END_EPIWEEK = 201920     # W20-2019, FluSight 2018-19 시즌 끝


# ============================================================================
# Forward + APMD decomposition per range
# ============================================================================
def _forward_dataset(model, ds, device) -> pd.DataFrame:
    """Run model on dataset, return per-(origin, horizon) (mu_z, s2_within_z, s2_between_z, y_z, target_ep)."""
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    rows = []
    H_list = HORIZONS
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            env = batch["env"].to(device)
            y_z = batch["y"].cpu().numpy()                                  # [B, H]
            target_eps = batch["target_epiweeks"]
            if isinstance(target_eps, torch.Tensor):
                tep = target_eps.cpu().numpy()
            elif isinstance(target_eps, (list, tuple)):
                arr = [e.cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
                       for e in target_eps]
                tep = np.stack(arr, axis=-1) if arr[0].ndim == 1 else np.array(arr).T
            else:
                tep = np.array(target_eps)
            pred, inter = model(x, env, return_intermediates=True)
            mu_z = pred.cpu().numpy()                                        # [B, H]
            gamma_all = inter["gamma_all"].cpu().numpy()                     # [B, max_h, K]
            B = mu_z.shape[0]
            for b in range(B):
                for hi, h in enumerate(H_list):
                    rows.append(dict(
                        target_ep=int(tep[b, hi]),
                        horizon=h,
                        mu_z=float(mu_z[b, hi]),
                        gamma_h=gamma_all[b, h - 1, :].tolist(),
                        y_z=float(y_z[b, hi]),
                    ))
    return pd.DataFrame(rows)


def _decompose_apmd(df_pred: pd.DataFrame, mu_k_zili, s2_k_zili,
                     target_mean, target_std) -> pd.DataFrame:
    """Add mu, s2_within, s2_between, s2_total (raw wILI units) + y_true."""
    out = df_pred.copy()
    sws, sbs, sts = [], [], []
    for _, r in df_pred.iterrows():
        g = np.array(r["gamma_h"])
        mu_hmm_z = float((g * mu_k_zili).sum())
        sw_z = float((g * s2_k_zili).sum())
        sb_z = float((g * (mu_k_zili - mu_hmm_z) ** 2).sum())
        sws.append(sw_z * (target_std ** 2))
        sbs.append(sb_z * (target_std ** 2))
        sts.append(max(sw_z + sb_z, 1e-12) * (target_std ** 2))
    out["s2_within"] = sws
    out["s2_between"] = sbs
    out["s2_total"] = sts
    out["mu"] = out["mu_z"] * target_std + target_mean
    out["y_true"] = out["y_z"] * target_std + target_mean
    out = out.drop(columns=["gamma_h", "mu_z", "y_z"])
    return out


# ============================================================================
def load_final_model(config_id, n_layers, d_model, seed, device):
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, n_layers=n_layers, d_model=d_model,
        data_csv=FINAL_CSV, norm_json=FINAL_NORM_JSON,
    )
    model = CGForecaster(cfg)
    # D.4 fix: per-seed HMM dispatch (학습 시 사용한 HMM 과 동일).
    # 안 하면 ckpt 의 phase_module._A/._means (per-seed) 와 APMD 분해 input mu_k/s2_k
    # (= hmm.means/covars) 가 다른 HMM → self-inconsistent → PC2-a/WIS/Cov95 측정 무의미.
    hmm_dir = Path(str(FINAL_HMM_TPL).format(seed=seed))
    hmm = load_fitted_hmm(hmm_dir)
    model.prepare_for_stage2(hmm)
    # γ.6 정정: paper headline ckpt = Stage 3 best.pt (m1_8_stage3_train)
    ck_path = _ROOT / "runs/m1_8_stage3_train" / f"e1_final_{config_id}_s{seed}_stage3" / "best.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model, hmm, cfg


# ============================================================================
def collect_national_predictions(config_id, n_layers, d_model, seed, split, device):
    """National held-out (split='val') or test (split='test') 위 forward."""
    model, hmm, cfg = load_final_model(config_id, n_layers, d_model, seed, device)
    df = load_dataset_csv(FINAL_CSV)
    norm = load_norm_params(FINAL_NORM_JSON)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std  = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64)
    s2_k = hmm.covars[:, 0, 0].astype(np.float64)
    ds = MultiHorizonDataset(df, split=split, lookback=cfg.lookback,
                              horizons=tuple(cfg.horizons), norm=norm)
    df_pred = _forward_dataset(model, ds, device)
    df_pred = _decompose_apmd(df_pred, mu_k, s2_k, target_mean, target_std)
    del model, hmm
    gc.collect(); torch.cuda.empty_cache()
    return df_pred


def collect_regional_predictions(config_id, n_layers, d_model, seed, region, device):
    """Regional zero-shot: 지역 wILI CSV + national env, test_strict 만."""
    model, hmm, cfg = load_final_model(config_id, n_layers, d_model, seed, device)
    df_reg = rsd._build_region_df(region)
    norm = load_norm_params(FINAL_NORM_JSON)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std  = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64)
    s2_k = hmm.covars[:, 0, 0].astype(np.float64)
    ds = MultiHorizonDataset(df_reg, split="test", lookback=cfg.lookback,
                              horizons=tuple(cfg.horizons), norm=norm)
    df_pred = _forward_dataset(model, ds, device)
    df_pred = _decompose_apmd(df_pred, mu_k, s2_k, target_mean, target_std)
    df_pred = df_pred[df_pred.target_ep >= TEST_STRICT_START_EPIWEEK].reset_index(drop=True)
    del model, hmm
    gc.collect(); torch.cuda.empty_cache()
    return df_pred


# ============================================================================
# Pool + dual report (raw + calibrated for test_strict)
# ============================================================================
def pool_5seed_eval(df_per_seed: dict, label: str) -> dict:
    """5-seed pair-level pool. label = 'held_out' or 'test_strict_nat'.
       MAE 도 같이 보고 (#1 헤드라인 전제)."""
    df_pool = pd.concat([df_per_seed[s] for s in SEEDS], ignore_index=True)
    cov_raw, wis_raw = eval_cov95_wis(
        df_pool.mu.to_numpy(),
        df_pool.s2_total.to_numpy(),
        df_pool.y_true.to_numpy(),
    )
    mae = float(np.abs(df_pool.mu.to_numpy() - df_pool.y_true.to_numpy()).mean())
    return dict(label=label, n_pooled=int(len(df_pool)),
                 raw=dict(cov95=cov_raw, wis=wis_raw, mae=mae),
                 _pool_df=df_pool)


def flusight_2018_19_slice(df_pool: pd.DataFrame) -> dict:
    """held-out pool 에서 FluSight 2018-19 시즌 (target_ep ≤ 201920) slice."""
    sub = df_pool[df_pool.target_ep <= FLUSIGHT_2018_19_END_EPIWEEK]
    if len(sub) == 0:
        return dict(n=0, cov95=None, wis=None, mae=None)
    cov, wis = eval_cov95_wis(sub.mu.to_numpy(), sub.s2_total.to_numpy(),
                                 sub.y_true.to_numpy())
    mae = float(np.abs(sub.mu.to_numpy() - sub.y_true.to_numpy()).mean())
    return dict(n=int(len(sub)),
                 unique_origins=int(sub.target_ep.nunique()),
                 cov95=cov, wis=wis, mae=mae)


def eval_dual_with_head(df_test_pool: pd.DataFrame,
                        df_cal_pool: pd.DataFrame) -> dict:
    """E3 head fit on df_cal_pool → apply on df_test_pool → dual (raw + calibrated)."""
    # 5-seed 평균 (target_ep, horizon) 단위 pivot — NaN 차단
    cal_avg = (df_cal_pool.groupby(["target_ep", "horizon"])
                .agg(mu=("mu", "mean"),
                     s2_within=("s2_within", "mean"),
                     s2_between=("s2_between", "mean"),
                     y_true=("y_true", "first"))
                .reset_index())
    cal_wide = cal_avg.pivot(index="target_ep", columns="horizon",
                              values=["mu", "s2_within", "s2_between", "y_true"]).dropna()
    mu_cal = cal_wide["mu"].to_numpy()                                       # [N_cal, H]
    sw_cal = cal_wide["s2_within"].to_numpy()
    sb_cal = cal_wide["s2_between"].to_numpy()
    y_cal  = cal_wide["y_true"].to_numpy()
    assert not (np.isnan(mu_cal).any() or np.isnan(sw_cal).any()
                or np.isnan(sb_cal).any() or np.isnan(y_cal).any()), \
        f"NaN in cal reshape: mu_cal {mu_cal.shape}, dropna 후에도 NaN — pivot 검토"
    print(f"     cal fit n_origin={len(mu_cal)}, horizons={mu_cal.shape[1]}", flush=True)
    head = fit_hetero_head(mu_cal, sw_cal, sb_cal, y_cal,
                             n_horizons=4, epochs=500, lr=1e-2, verbose=False)
    # Apply on test (kept long format, easy)
    mu_te = df_test_pool.mu.to_numpy()
    sw_te = df_test_pool.s2_within.to_numpy()
    sb_te = df_test_pool.s2_between.to_numpy()
    y_te  = df_test_pool.y_true.to_numpy()
    s2_raw = sw_te + sb_te
    cov_raw, wis_raw = eval_cov95_wis(mu_te, s2_raw, y_te)
    # Apply head per-row (single horizon at a time would be cleaner; head is per-h)
    h_arr = df_test_pool.horizon.to_numpy()
    s2_cal = np.empty_like(s2_raw)
    params = head.get_params()
    alpha = np.array(params["alpha"])
    beta = np.array(params["beta"])
    for h in HORIZONS:
        idx = (h_arr == h)
        h_pos = h - 1
        s2_cal[idx] = alpha[h_pos] * sw_te[idx] + beta[h_pos] * sb_te[idx]
    cov_cal, wis_cal = eval_cov95_wis(mu_te, s2_cal, y_te)
    return dict(
        raw=dict(cov95=cov_raw, wis=wis_raw),
        calibrated=dict(cov95=cov_cal, wis=wis_cal,
                          alpha=params["alpha"], beta=params["beta"]),
        head_params=params,
    )


# ============================================================================
# PC2-a 재확인 (HEADLINE_CONFIG_ID 만, regional test_strict, phase-anchored vs s_h scalar)
# ============================================================================
def pc2a_recheck(df_per_region_per_seed: dict, val_residuals_per_h: dict) -> dict:
    """phase-anchored 분산 vs s_h scalar (quantile-matching on held-out residuals)."""
    # s_h fit on val period (national) — paper PC2-a 와 동일 패턴 (regime_shift_drivers 의 cov95_wis)
    PC2A_SH_GRID = np.concatenate([
        np.linspace(0.05, 0.5, 30), np.linspace(0.5, 2.0, 30),
        np.linspace(2.0, 5.0, 15),
    ])
    s_h_per_h = []
    for h in HORIZONS:
        # val_residuals_per_h[h] = list of (mu, y) tuples from held-out national
        resid = val_residuals_per_h[h]
        mu_v = np.array([t[0] for t in resid])
        y_v  = np.array([t[1] for t in resid])
        # Grid search
        best_loss, best_s = np.inf, None
        for s in PC2A_SH_GRID:
            err = 0.0
            for tau in FLUSIGHT_23:
                z = sp_norm.ppf(tau)
                q_pred = mu_v + z * s
                emp = float((y_v <= q_pred).mean())
                err += (emp - tau) ** 2
            if err < best_loss:
                best_loss, best_s = err, float(s)
        s_h_per_h.append(best_s)

    # Per-region (phase Cov95, s_h Cov95, ΔCov95 etc.)
    per_region = {}
    for region, df_seeds in df_per_region_per_seed.items():
        df_pool = pd.concat([df_seeds[s] for s in SEEDS], ignore_index=True)
        # phase-anchored
        cov_p, wis_p = eval_cov95_wis(df_pool.mu.to_numpy(),
                                         df_pool.s2_total.to_numpy(),
                                         df_pool.y_true.to_numpy())
        # s_h scalar (constant per horizon)
        mu_te = df_pool.mu.to_numpy(); y_te = df_pool.y_true.to_numpy()
        h_arr = df_pool.horizon.to_numpy()
        s2_sh = np.empty_like(mu_te)
        for h in HORIZONS:
            s2_sh[h_arr == h] = s_h_per_h[h - 1] ** 2
        cov_s, wis_s = eval_cov95_wis(mu_te, s2_sh, y_te)
        per_region[region] = dict(cov_phase=cov_p, wis_phase=wis_p,
                                     cov_sh=cov_s, wis_sh=wis_s,
                                     dcov=cov_p - cov_s, dwis=wis_p - wis_s)

    cov_p_per_r = {r: per_region[r]["cov_phase"] for r in REGIONS_TRANSFER}
    cov_s_per_r = {r: per_region[r]["cov_sh"] for r in REGIONS_TRANSFER}
    wis_p_per_r = {r: per_region[r]["wis_phase"] for r in REGIONS_TRANSFER}
    wis_s_per_r = {r: per_region[r]["wis_sh"] for r in REGIONS_TRANSFER}
    dcov_m, dcov_lo, dcov_hi = cluster_bootstrap_delta(cov_p_per_r, cov_s_per_r)
    dwis_m, dwis_lo, dwis_hi = cluster_bootstrap_delta(wis_p_per_r, wis_s_per_r)
    return dict(
        s_h_per_horizon=s_h_per_h,
        per_region=per_region,
        bootstrap=dict(
            dcov=dict(mean=dcov_m, lo=dcov_lo, hi=dcov_hi,
                       excludes_0=(dcov_lo > 0 or dcov_hi < 0)),
            dwis=dict(mean=dwis_m, lo=dwis_lo, hi=dwis_hi,
                       excludes_0=(dwis_lo > 0 or dwis_hi < 0)),
        ),
    )


# ============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aggregate = dict(
        cal_set_lock=dict(
            test_strict_eval_fit_on="held-out (W40-2018 ~ W10-2020)",
            flusight_eval_fit="raw-only (head 적용 안 함)",
            pc2a_eval="raw HMM 분산 only (γ.5 narrative lock)",
        ),
    )

    for config_id, n_layers, d_model in CONFIGS:
        print(f"\n{'='*80}\n[FINAL EVAL] config={config_id}\n{'='*80}", flush=True)
        nat_holdout, nat_teststrict = {}, {}
        reg_per_region_per_seed = {r: {} for r in REGIONS_TRANSFER}

        for seed in SEEDS:
            print(f"  seed {seed}: collecting national held-out + test_strict ...", flush=True)
            nat_holdout[seed] = collect_national_predictions(
                config_id, n_layers, d_model, seed, split="val", device=device)
            df_test_all = collect_national_predictions(
                config_id, n_layers, d_model, seed, split="test", device=device)
            nat_teststrict[seed] = df_test_all[df_test_all.target_ep >= TEST_STRICT_START_EPIWEEK
                                                ].reset_index(drop=True)
            print(f"     held-out n={len(nat_holdout[seed])}  test_strict_nat n={len(nat_teststrict[seed])}", flush=True)

            if config_id == HEADLINE_CONFIG_ID:
                for region in REGIONS_TRANSFER:
                    reg_per_region_per_seed[region][seed] = collect_regional_predictions(
                        config_id, n_layers, d_model, seed, region, device=device)
                print(f"     regional 10 done", flush=True)

        # Pool 5-seed
        ho_pool = pool_5seed_eval(nat_holdout, "held_out_national_raw_only")
        ts_pool = pool_5seed_eval(nat_teststrict, "test_strict_national")
        # MAE 추가 (test_strict national raw, calibrated 후에도 mu 동일이라 MAE 무관)
        ts_mae = float(np.abs(ts_pool["_pool_df"].mu - ts_pool["_pool_df"].y_true).mean())
        # FluSight 2018-19 시즌만 슬라이스 (held-out subset)
        fs_slice = flusight_2018_19_slice(ho_pool["_pool_df"])
        # E3 dual on test_strict (cal=held-out)
        ts_dual = eval_dual_with_head(ts_pool["_pool_df"], ho_pool["_pool_df"])
        # per-origin parquet 저장 (audit trail + 미래 분석)
        ho_pool["_pool_df"].to_parquet(OUT_DIR / f"{config_id}_held_out_national.parquet", index=False)
        ts_pool["_pool_df"].to_parquet(OUT_DIR / f"{config_id}_test_strict_national.parquet", index=False)

        role = "HEADLINE (사전등록 누설-free winner)" if config_id == HEADLINE_CONFIG_ID else "ROBUSTNESS CHECK (paper-config)"
        print(f"\n  ── {config_id} [{role}] ──")
        print(f"  held-out (full val period W40-2018~W10-2020, raw): "
              f"Cov95={ho_pool['raw']['cov95']:.4f}  WIS={ho_pool['raw']['wis']:.4f}  "
              f"MAE={ho_pool['raw']['mae']:.4f}  n={ho_pool['n_pooled']}")
        print(f"  FluSight 2018-19 시즌만 (W40-2018~W20-2019, raw): "
              f"Cov95={fs_slice['cov95']:.4f}  WIS={fs_slice['wis']:.4f}  "
              f"MAE={fs_slice['mae']:.4f}  n={fs_slice['n']} (unique_origins={fs_slice['unique_origins']})")
        print(f"  test_strict national RAW :  Cov95={ts_dual['raw']['cov95']:.4f}  "
              f"WIS={ts_dual['raw']['wis']:.4f}  MAE={ts_mae:.4f}")
        print(f"  test_strict national CAL :  Cov95={ts_dual['calibrated']['cov95']:.4f}  "
              f"WIS={ts_dual['calibrated']['wis']:.4f}  (MAE 동일 {ts_mae:.4f})")
        print(f"     head α={[f'{a:.3f}' for a in ts_dual['head_params']['alpha']]}  "
              f"β={[f'{b:.3f}' for b in ts_dual['head_params']['beta']]}")

        aggregate[config_id] = dict(
            held_out_national=ho_pool['raw'],
            held_out_n_pooled=ho_pool['n_pooled'],
            flusight_2018_19_season=fs_slice,
            test_strict_national=dict(**ts_dual, mae=ts_mae,
                                         n_pooled=int(len(ts_pool['_pool_df']))),
        )

        # Regional + PC2-a (HEADLINE_CONFIG_ID = 사전등록 누설-free winner only)
        if config_id == HEADLINE_CONFIG_ID:
            # test_strict regional pool per region (5-seed)
            reg_raw, reg_dual = {}, {}
            ho_pool_for_reg_cal = ho_pool["_pool_df"]                       # cal-set = held-out
            for region in REGIONS_TRANSFER:
                pool_r = pd.concat([reg_per_region_per_seed[region][s] for s in SEEDS],
                                     ignore_index=True)
                # Raw
                cov_r, wis_r = eval_cov95_wis(pool_r.mu.to_numpy(),
                                                 pool_r.s2_total.to_numpy(),
                                                 pool_r.y_true.to_numpy())
                reg_raw[region] = dict(cov95=cov_r, wis=wis_r, n=int(len(pool_r)))
                # Calibrated: same head trained on held-out (transfer-aware)
                # (Region-by-region cal fit 은 in-sample 위험 → 동일 head 적용)
            # Apply head to all regional pools (head from held-out cal-set, 5-seed mean pivot)
            cal_avg_reg = (ho_pool_for_reg_cal.groupby(["target_ep", "horizon"])
                            .agg(mu=("mu", "mean"),
                                 s2_within=("s2_within", "mean"),
                                 s2_between=("s2_between", "mean"),
                                 y_true=("y_true", "first"))
                            .reset_index())
            mu_cal_w = cal_avg_reg.pivot(index="target_ep", columns="horizon",
                                            values=["mu","s2_within","s2_between","y_true"]).dropna()
            head_reg = fit_hetero_head(
                mu_cal_w["mu"].to_numpy(), mu_cal_w["s2_within"].to_numpy(),
                mu_cal_w["s2_between"].to_numpy(), mu_cal_w["y_true"].to_numpy(),
                n_horizons=4, epochs=500, lr=1e-2, verbose=False)
            params_reg = head_reg.get_params()
            for region in REGIONS_TRANSFER:
                pool_r = pd.concat([reg_per_region_per_seed[region][s] for s in SEEDS],
                                     ignore_index=True)
                mu_te = pool_r.mu.to_numpy(); y_te = pool_r.y_true.to_numpy()
                sw_te = pool_r.s2_within.to_numpy(); sb_te = pool_r.s2_between.to_numpy()
                h_arr = pool_r.horizon.to_numpy()
                alpha = np.array(params_reg["alpha"]); beta = np.array(params_reg["beta"])
                s2_cal = np.empty_like(mu_te)
                for h in HORIZONS:
                    idx = h_arr == h
                    s2_cal[idx] = alpha[h - 1] * sw_te[idx] + beta[h - 1] * sb_te[idx]
                cov_c, wis_c = eval_cov95_wis(mu_te, s2_cal, y_te)
                reg_dual[region] = dict(raw=reg_raw[region],
                                          calibrated=dict(cov95=cov_c, wis=wis_c))

            # Cluster bootstrap on regional Cov95 (raw and calibrated)
            cov_raw_per_r = {r: reg_raw[r]["cov95"] for r in REGIONS_TRANSFER}
            cov_cal_per_r = {r: reg_dual[r]["calibrated"]["cov95"] for r in REGIONS_TRANSFER}
            cov_raw_m, cov_raw_lo, cov_raw_hi = cluster_bootstrap_mean(cov_raw_per_r)
            cov_cal_m, cov_cal_lo, cov_cal_hi = cluster_bootstrap_mean(cov_cal_per_r)
            wis_raw_per_r = {r: reg_raw[r]["wis"] for r in REGIONS_TRANSFER}
            wis_cal_per_r = {r: reg_dual[r]["calibrated"]["wis"] for r in REGIONS_TRANSFER}
            wis_raw_m, wis_raw_lo, wis_raw_hi = cluster_bootstrap_mean(wis_raw_per_r)
            wis_cal_m, wis_cal_lo, wis_cal_hi = cluster_bootstrap_mean(wis_cal_per_r)
            # Regional MAE (per region 평균 후 cluster bootstrap)
            mae_raw_per_r = {}
            for region in REGIONS_TRANSFER:
                pool_r = pd.concat([reg_per_region_per_seed[region][s] for s in SEEDS],
                                     ignore_index=True)
                mae_raw_per_r[region] = float(np.abs(pool_r.mu - pool_r.y_true).mean())
                pool_r.to_parquet(OUT_DIR / f"{config_id}_test_strict_{region}.parquet",
                                   index=False)
            mae_raw_m, mae_raw_lo, mae_raw_hi = cluster_bootstrap_mean(mae_raw_per_r)
            print(f"\n  ── {config_id} REGIONAL TRANSFER ──")
            print(f"  cross-region RAW Cov95: {cov_raw_m:.4f}  CI[{cov_raw_lo:.4f}, {cov_raw_hi:.4f}]")
            print(f"  cross-region CAL Cov95: {cov_cal_m:.4f}  CI[{cov_cal_lo:.4f}, {cov_cal_hi:.4f}]")
            print(f"  cross-region RAW WIS  : {wis_raw_m:.4f}  CI[{wis_raw_lo:.4f}, {wis_raw_hi:.4f}]")
            print(f"  cross-region CAL WIS  : {wis_cal_m:.4f}  CI[{wis_cal_lo:.4f}, {wis_cal_hi:.4f}]")
            print(f"  cross-region RAW MAE  : {mae_raw_m:.4f}  CI[{mae_raw_lo:.4f}, {mae_raw_hi:.4f}]")

            aggregate["regional_transfer"] = dict(
                per_region=reg_dual,
                per_region_mae=mae_raw_per_r,
                bootstrap_raw=dict(
                    cov95=dict(mean=cov_raw_m, lo=cov_raw_lo, hi=cov_raw_hi),
                    wis=dict(mean=wis_raw_m, lo=wis_raw_lo, hi=wis_raw_hi),
                    mae=dict(mean=mae_raw_m, lo=mae_raw_lo, hi=mae_raw_hi)),
                bootstrap_cal=dict(
                    cov95=dict(mean=cov_cal_m, lo=cov_cal_lo, hi=cov_cal_hi),
                    wis=dict(mean=wis_cal_m, lo=wis_cal_lo, hi=wis_cal_hi)),
            )

            # PC2-a 재확인 (val residuals from held-out 5-seed)
            print(f"\n  ── {config_id} PC2-a RECHECK (phase-anchored vs s_h) ──")
            val_resid = {}
            for h in HORIZONS:
                pairs = []
                for s in SEEDS:
                    sub = nat_holdout[s][nat_holdout[s].horizon == h]
                    for _, row in sub.iterrows():
                        pairs.append((row.mu, row.y_true))
                val_resid[h] = pairs
            pc2a = pc2a_recheck(reg_per_region_per_seed, val_resid)
            bs = pc2a["bootstrap"]
            print(f"  ΔCov95 (phase-s_h): {bs['dcov']['mean']:+.4f}  "
                  f"CI[{bs['dcov']['lo']:+.4f}, {bs['dcov']['hi']:+.4f}]  excludes_0={bs['dcov']['excludes_0']}")
            print(f"  ΔWIS   (phase-s_h): {bs['dwis']['mean']:+.4f}  "
                  f"CI[{bs['dwis']['lo']:+.4f}, {bs['dwis']['hi']:+.4f}]  excludes_0={bs['dwis']['excludes_0']}")
            aggregate[f"pc2a_recheck_{HEADLINE_CONFIG_ID}"] = pc2a

    # Drop _pool_df before save
    out_path = OUT_DIR / "e1_final_eval.json"
    aggregate_clean = json.loads(json.dumps(aggregate, default=str))
    with open(out_path, "w") as f:
        json.dump(aggregate_clean, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
