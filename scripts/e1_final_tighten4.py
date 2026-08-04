"""scripts/e1_final_tighten4.py — Tightening 4 (gate before §V writing)

4-1: cal-head leak-free 검증 (별도 처리, 4-1 PASS — train.max=201839 < val.min=201840)
4-2: leaky-cal 추출 (paper ablation_retrain.full ckpt × 5 seeds × 2 splits forward
                     → cal-head fit on held-out → apply on test_strict
                     → apples-to-apples clean-cal vs leaky-cal)
4-3: 불확실성 통일표 (raw/cal × national/regional × phase/s_h × clean/leaky)
"""
from __future__ import annotations
import dataclasses
import gc
import json
import math
import statistics as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as sp_norm
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.models.cg_forecaster import CGForecaster
from src.models.heteroscedastic_head import fit_hetero_head, eval_cov95_wis, FLUSIGHT_23
from src.utils.checkpoints import load_fitted_hmm
from src.utils.config import CGMambaConfig
from src.data.loader import load_dataset_csv, load_norm_params, MultiHorizonDataset

FINAL_CSV       = _ROOT / "data/processed/ili_env_weekly_split.csv"
FINAL_NORM_JSON = _ROOT / "data/processed/normalization_params.json"
FINAL_HMM_TPL   = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"
OUT_DIR         = _ROOT / "runs/e1_final"

SEEDS = [42, 123, 456, 789, 1024]
HORIZONS = [1, 2, 3, 4]
TEST_STRICT_START_EPIWEEK = 202240


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va/na + vb/nb)
    t = (ma - mb) / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return ma - mb, t, p


def _forward_dataset(model, ds, device):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device); env = batch["env"].to(device)
            y_z = batch["y"].cpu().numpy()
            target_eps = batch["target_epiweeks"]
            if isinstance(target_eps, torch.Tensor):
                tep = target_eps.cpu().numpy()
            elif isinstance(target_eps, (list, tuple)):
                arr = [t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                       for t in target_eps]
                tep = np.stack(arr, axis=-1) if arr[0].ndim == 1 else np.array(arr).T
            else:
                tep = np.array(target_eps)
            pred, inter = model(x, env, return_intermediates=True)
            mu_z = pred.cpu().numpy()
            gamma_all = inter["gamma_all"].cpu().numpy()
            B, H = mu_z.shape
            for b in range(B):
                for hi in range(H):
                    rows.append(dict(target_ep=int(tep[b, hi]), horizon=hi+1,
                                       gamma_h=gamma_all[b, hi, :], mu_z=float(mu_z[b, hi]),
                                       y_z=float(y_z[b, hi])))
    return pd.DataFrame(rows)


def _decompose_apmd(df, mu_k, s2_k, target_mean, target_std):
    gamma_h = np.stack(df.gamma_h.values)
    mu_hmm = (gamma_h * mu_k).sum(1)
    sw_z = (gamma_h * s2_k).sum(1)
    sb_z = (gamma_h * (mu_k - mu_hmm[:, None])**2).sum(1)
    df = df.copy()
    df["mu"] = df.mu_z * target_std + target_mean
    df["y_true"] = df.y_z * target_std + target_mean
    df["s2_within"] = sw_z * target_std**2
    df["s2_between"] = sb_z * target_std**2
    df["s2_total"] = df.s2_within + df.s2_between
    return df.drop(columns=["gamma_h", "mu_z", "y_z"])


def collect_paper_predictions(seed, split, device):
    """paper ablation_retrain.full ckpt × seed × split forward + APMD 분해."""
    cfg = dataclasses.replace(
        CGMambaConfig(),
        seed=seed, n_layers=3, d_model=64,
        data_csv=FINAL_CSV, norm_json=FINAL_NORM_JSON,
        # paper CG_TOP1_HP override (학습 시 사용된 그대로)
        dropout=0.0, lookback=104,
        stage2_gate_lr=1e-3, stage2_backbone_lr=1e-4,
        stage3_other_lr=1e-4, stage3_hmm_lr=1e-6,
        stage3_state_embed_lr=1e-6, stage3_env_lr=1e-7,
    )
    model = CGForecaster(cfg)
    hmm_dir = Path(str(FINAL_HMM_TPL).format(seed=seed))   # per-seed (D.4 self-consistent)
    hmm = load_fitted_hmm(hmm_dir)
    model.prepare_for_stage2(hmm)
    ck_path = _ROOT / f"runs/m1_8_stage3_train/ablation_retrain_full_s{seed}_stage3/best.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    df_data = load_dataset_csv(FINAL_CSV)
    norm = load_norm_params(FINAL_NORM_JSON)
    target_mean = float(norm["ili_weighted_pct"]["mean"])
    target_std  = float(norm["ili_weighted_pct"]["std"])
    mu_k = hmm.means[:, 0].astype(np.float64)
    s2_k = hmm.covars[:, 0, 0].astype(np.float64)
    ds = MultiHorizonDataset(df_data, split=split, lookback=cfg.lookback,
                              horizons=tuple(cfg.horizons), norm=norm)
    df_pred = _forward_dataset(model, ds, device)
    df_pred = _decompose_apmd(df_pred, mu_k, s2_k, target_mean, target_std)
    del model, hmm
    gc.collect(); torch.cuda.empty_cache()
    return df_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("# Tightening 4 — cal-head leak-free + leaky-cal apples-to-apples")
    print("=" * 80)

    print("\n## 4-1: cal-head leak-free 검증")
    print("  ✓ train.max_epiweek = 201839")
    print("  ✓ val.min_epiweek   = 201840  (cal-set)")
    print("  ✓ test.min_epiweek  = 202040")
    print("  ✓ time order: train < val(cal) < covid_excluded < test → leak-free")

    print("\n## 4-2: paper leaky ckpt × 5 seeds × (held-out + test_strict) forward")
    paper_holdout, paper_teststrict = {}, {}
    for seed in SEEDS:
        print(f"  seed {seed}: forwarding paper leaky ckpt ...", flush=True)
        paper_holdout[seed] = collect_paper_predictions(seed, "val", device)
        df_test = collect_paper_predictions(seed, "test", device)
        paper_teststrict[seed] = df_test[df_test.target_ep >= TEST_STRICT_START_EPIWEEK
                                          ].reset_index(drop=True)
        print(f"     held-out n={len(paper_holdout[seed])}  "
              f"test_strict n={len(paper_teststrict[seed])}", flush=True)

    # Pool 5-seed cal-set (= held-out)
    ho_pool = pd.concat([paper_holdout[s] for s in SEEDS], ignore_index=True)
    ts_pool = pd.concat([paper_teststrict[s] for s in SEEDS], ignore_index=True)

    # Fit cal-head on held-out (paper-cal, 5-seed mean pivot)
    cal_avg = (ho_pool.groupby(["target_ep", "horizon"])
                .agg(mu=("mu", "mean"),
                     s2_within=("s2_within", "mean"),
                     s2_between=("s2_between", "mean"),
                     y_true=("y_true", "first"))
                .reset_index())
    cal_wide = cal_avg.pivot(index="target_ep", columns="horizon",
                              values=["mu", "s2_within", "s2_between", "y_true"]).dropna()
    print(f"  cal fit (paper leaky): n_origin={len(cal_wide)}, horizons={cal_wide['mu'].shape[1]}")
    head = fit_hetero_head(cal_wide["mu"].to_numpy(),
                            cal_wide["s2_within"].to_numpy(),
                            cal_wide["s2_between"].to_numpy(),
                            cal_wide["y_true"].to_numpy(),
                            n_horizons=4, epochs=500, lr=1e-2, verbose=False)
    params = head.get_params()
    alpha = np.array(params["alpha"]); beta = np.array(params["beta"])
    print(f"  paper-cal head α={[f'{a:.3f}' for a in params['alpha']]}  "
          f"β={[f'{b:.3f}' for b in params['beta']]}")

    # Per-seed apples-to-apples: paper-raw + paper-cal (leaky) vs clean (이미 측정됨)
    print("\n## leaky paper-raw + leaky paper-cal per-seed (test_strict national)")
    leaky_raw_per_seed, leaky_cal_per_seed = [], []
    for s in SEEDS:
        df_s = paper_teststrict[s]
        mu = df_s.mu.to_numpy()
        y  = df_s.y_true.to_numpy()
        sw = df_s.s2_within.to_numpy(); sb = df_s.s2_between.to_numpy()
        s2_raw = sw + sb
        s2_cal = np.empty_like(s2_raw)
        h_arr = df_s.horizon.to_numpy()
        for h in HORIZONS:
            idx = h_arr == h
            s2_cal[idx] = alpha[h-1] * sw[idx] + beta[h-1] * sb[idx]
        cov_raw, wis_raw = eval_cov95_wis(mu, s2_raw, y)
        cov_cal, wis_cal = eval_cov95_wis(mu, s2_cal, y)
        mae = float(np.abs(mu - y).mean())
        leaky_raw_per_seed.append(dict(seed=s, mae=mae, wis=wis_raw, cov95=cov_raw))
        leaky_cal_per_seed.append(dict(seed=s, mae=mae, wis=wis_cal, cov95=cov_cal))
        print(f"  s{s}: paper RAW cov={cov_raw:.4f} wis={wis_raw:.4f}  |  "
              f"paper CAL cov={cov_cal:.4f} wis={wis_cal:.4f}  MAE={mae:.4f}")

    # Pull clean (n3_d64) per-seed from e1_final_tightening.json
    clean_per_seed = json.load(open(OUT_DIR / "e1_final_tightening.json"))[
        "tightening_3_clean_vs_leaky_n3_d64"]["clean_per_seed"]
    # clean_per_seed 의 wis/cov 는 raw — clean cal 도 같이 계산해야 (e1_final_eval.json 에서 확인)
    clean_eval = json.load(open(OUT_DIR / "e1_final_eval.json"))
    clean_n3d64_test = clean_eval["n3_d64"]["test_strict_national"]
    print(f"\n  clean n3_d64 5-seed pooled RAW: cov={clean_n3d64_test['raw']['cov95']:.4f} "
          f"wis={clean_n3d64_test['raw']['wis']:.4f}")
    print(f"  clean n3_d64 5-seed pooled CAL: cov={clean_n3d64_test['calibrated']['cov95']:.4f} "
          f"wis={clean_n3d64_test['calibrated']['wis']:.4f}")

    # Per-seed apples-to-apples (Welch): clean-cal vs leaky-cal
    # clean per-seed cal 측정 위해 parquet 재집계 필요 (clean 의 cal head 도 같이 가)
    print("\n## clean n3_d64 per-seed cal 측정 (e1_final 의 clean cal head 재현)")
    clean_eval_full = json.load(open(OUT_DIR / "e1_final_eval.json"))
    # clean cal head params on e1_final_eval (already fit on clean held-out)
    clean_cal_alpha = np.array(clean_eval_full["n3_d64"]["test_strict_national"]["calibrated"]["alpha"])
    clean_cal_beta  = np.array(clean_eval_full["n3_d64"]["test_strict_national"]["calibrated"]["beta"])
    clean_pq = pd.read_parquet(OUT_DIR / "n3_d64_test_strict_national.parquet")
    n_per = len(clean_pq) // len(SEEDS)
    clean_raw_per, clean_cal_per = [], []
    for i, s in enumerate(SEEDS):
        df_s = clean_pq.iloc[i*n_per:(i+1)*n_per].reset_index(drop=True)
        mu = df_s.mu.to_numpy(); y = df_s.y_true.to_numpy()
        sw = df_s.s2_within.to_numpy(); sb = df_s.s2_between.to_numpy()
        s2_raw = sw + sb
        s2_cal = np.empty_like(s2_raw)
        h_arr = df_s.horizon.to_numpy()
        for h in HORIZONS:
            idx = h_arr == h
            s2_cal[idx] = clean_cal_alpha[h-1] * sw[idx] + clean_cal_beta[h-1] * sb[idx]
        cov_raw, wis_raw = eval_cov95_wis(mu, s2_raw, y)
        cov_cal, wis_cal = eval_cov95_wis(mu, s2_cal, y)
        clean_raw_per.append(dict(seed=s, wis=wis_raw, cov95=cov_raw))
        clean_cal_per.append(dict(seed=s, wis=wis_cal, cov95=cov_cal))

    print("\n## 4-2 결과: apples-to-apples Welch (RAW vs RAW, CAL vs CAL)")
    print(f"\n  {'comparison':>32s}  {'leaky_paper':>13s}  {'clean':>13s}  {'Δ':>9s}  {'p':>7s}")
    for kind, leaky_list, clean_list in [
        ("RAW Cov95 (paper vs clean)",
         [r["cov95"] for r in leaky_raw_per_seed],
         [r["cov95"] for r in clean_raw_per]),
        ("RAW WIS   (paper vs clean)",
         [r["wis"]   for r in leaky_raw_per_seed],
         [r["wis"]   for r in clean_raw_per]),
        ("CAL Cov95 (paper vs clean) ← gate",
         [r["cov95"] for r in leaky_cal_per_seed],
         [r["cov95"] for r in clean_cal_per]),
        ("CAL WIS   (paper vs clean) ← gate",
         [r["wis"]   for r in leaky_cal_per_seed],
         [r["wis"]   for r in clean_cal_per]),
    ]:
        d, t, p = welch_t(leaky_list, clean_list)
        ml, mc = st.mean(leaky_list), st.mean(clean_list)
        sl, sc = st.stdev(leaky_list), st.stdev(clean_list)
        print(f"  {kind:>32s}  {ml:.4f}±{sl:.4f}  {mc:.4f}±{sc:.4f}  "
              f"{d:+.4f}  {p:.3f}")

    # Tightening 4-3: 통일표
    print("\n## 4-3: 불확실성 통일표 (Cov95)")
    print(f"\n  {'setting':<32s}  {'national raw':>13s}  {'national cal':>13s}  "
          f"{'regional phase':>15s}  {'regional s_h':>13s}")
    print(f"  {'paper leaky n3_d64':<32s}  "
          f"{st.mean([r['cov95'] for r in leaky_raw_per_seed]):>13.4f}  "
          f"{st.mean([r['cov95'] for r in leaky_cal_per_seed]):>13.4f}  "
          f"{'N/A':>15s}  {'N/A':>13s}")
    print(f"  {'clean n2_d128 (headline)':<32s}  "
          f"{clean_eval_full['n2_d128']['test_strict_national']['raw']['cov95']:>13.4f}  "
          f"{clean_eval_full['n2_d128']['test_strict_national']['calibrated']['cov95']:>13.4f}  "
          f"{clean_eval_full['regional_transfer']['bootstrap_raw']['cov95']['mean']:>15.4f}  "
          f"{'N/A':>13s}")
    print(f"  {'clean n3_d64 (robustness)':<32s}  "
          f"{clean_eval_full['n3_d64']['test_strict_national']['raw']['cov95']:>13.4f}  "
          f"{clean_eval_full['n3_d64']['test_strict_national']['calibrated']['cov95']:>13.4f}  "
          f"{'N/A':>15s}  {'N/A':>13s}")
    # PC2-a phase vs s_h (n2_d128, regional)
    pc2a = clean_eval_full["pc2a_recheck_n2_d128"]
    cov_p = st.mean([pc2a["per_region"][r]["cov_phase"] for r in pc2a["per_region"]])
    cov_s = st.mean([pc2a["per_region"][r]["cov_sh"] for r in pc2a["per_region"]])
    print(f"  {'phase-anchored regional (n2_d128)':<32s}  {'N/A':>13s}  {'N/A':>13s}  "
          f"{cov_p:>15.4f}  {cov_s:>13.4f}")

    # Save
    out = {
        "tightening_4_1_cal_head_leak_free": {
            "train_max": 201839, "val_min": 201840, "test_min": 202040,
            "verdict": "leak-free (train < val < test, time-ordered)",
        },
        "tightening_4_2_leaky_cal_extracted": {
            "paper_cal_head_alpha": params["alpha"],
            "paper_cal_head_beta":  params["beta"],
            "leaky_raw_per_seed": leaky_raw_per_seed,
            "leaky_cal_per_seed": leaky_cal_per_seed,
            "clean_raw_per_seed": clean_raw_per,
            "clean_cal_per_seed": clean_cal_per,
        },
    }
    with (OUT_DIR / "e1_final_tighten4.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: runs/e1_final/e1_final_tighten4.json")


if __name__ == "__main__":
    main()
