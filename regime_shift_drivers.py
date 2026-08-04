"""regime_shift_drivers.py

Phase 0 audit가 막은 3.5 fields(1·5·6·8)를 푸는 드라이버 2종 + load_data 빌더.

발견된 자산에 맞춤:
  - 출력 스키마: runs/compare_baselines/baseline_predictions_seed42.csv
        (baseline, horizon, target_ep, mu, sigma, y_true) + APMD 컬럼 확장
  - CGM 5-seed: runs/m2_4_data_efficiency/cg_mamba/seasons_17_seasons_full/seed{S}/manifest.json
  - Vanilla 5-seed: runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{S}/vanilla_mamba_best.pt
  - no_encgates 5-seed: runs/m1_7_train/ablation_retrain_no_encgates_s{S}_stage2/best.pt
  - frozen HMM: runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{S}/
  - forward-backward: src/models/phase_module.py::_torch_forward_backward
  - APMD: src/eval/hmm_interval.py::compute_decomposition (gamma_all + HMM emission stats)

실행:
  1) python regime_shift_drivers.py dump      # fields 1·5·8
  2) python regime_shift_drivers.py gamma     # field 6
  3) regime_shift_experiment.py 의 load_data() 를 build_load_data() 로 교체
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

# 1st-party imports (eval scripts 재활용)
from baselines.vanilla_mamba import VanillaMambaForecaster          # noqa
from src.models.cg_forecaster import CGForecaster                   # noqa
from src.utils.checkpoints import load_fitted_hmm                   # noqa
from src.utils.config import CGMambaConfig                          # noqa
from src.data.loader import WeeklyDataset, load_norm_params         # noqa
from src.baselines.lstm import WeeklyMultiHorizonDataset            # noqa
from src.eval.hmm_interval import compute_decomposition             # noqa

OUT_DIR  = _ROOT / "runs/regime_shift"
SEEDS    = [42, 123, 456, 789, 1024]
REGIONS  = ["national"] + [f"hhs{i}" for i in range(1, 11)]
MODELS   = ["cg_mamba", "vanilla_mamba", "no_encgates"]              # 핵심 3 arm
MODELS_PLUS_NOENV = MODELS + ["no_env"]                              # phase isolation arm 포함
HORIZONS = [1, 2, 3, 4]
SPLITS   = ["val", "test_strict"]                                    # val→field5, test→평가
HMM_TPL  = _ROOT / "runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}"

# Paper Table I/III/VI reference (national 5-seed test_strict, ablation_retrain stage3 best.pt)
PAPER_REF_MAE_AVG = {                                                # ablation_retrain_aggregate.csv
    "cg_mamba":      0.3904,   # = "full" 5-seed mean
    "no_encgates":   0.4922,
    "no_env":        0.5529,
}
DATA_DIR = _ROOT / "data"
NORM_JSON = DATA_DIR / "processed/normalization_params.json"
SPLIT_CSV = DATA_DIR / "processed/ili_env_weekly_split.csv"
REGION_CSV_TPL = DATA_DIR / "raw/cdc_ilinet/_phase3_phase6_fetch/{region}_full.csv"
ENV_NATIONAL_CSV = DATA_DIR / "processed/env_national_weekly.csv"
TS_BOUNDARY = 202240

NORM = load_norm_params(NORM_JSON)
TARGET_MEAN = float(NORM["ili_weighted_pct"]["mean"])
TARGET_STD  = float(NORM["ili_weighted_pct"]["std"])

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Region dataframe builder (phase_3_region_eval.py 패턴 그대로)
# ============================================================================
def _build_region_df(region: str) -> pd.DataFrame:
    if region == "national":
        return pd.read_csv(SPLIT_CSV)
    from epiweeks import Week
    df_r = pd.read_csv(str(REGION_CSV_TPL).format(region=region))
    df_r["epiweek"] = df_r["year"].astype(int) * 100 + df_r["week"].astype(int)
    df_r["date"] = df_r.apply(
        lambda r: Week(int(r["year"]), int(r["week"])).startdate().isoformat(), axis=1)
    env = pd.read_csv(ENV_NATIONAL_CSV)[["epiweek", "temperature_c", "specific_humidity_g_per_kg"]]
    split = pd.read_csv(SPLIT_CSV)[["epiweek", "split"]]
    df = df_r.merge(env, on="epiweek", how="inner").merge(split, on="epiweek", how="inner")
    df["n_stations_available"] = 10
    df["weight_sum_raw"] = 1.0
    return df


# ============================================================================
# HOOK 1 — checkpoint resolution
# ============================================================================
def _resolve_checkpoint(model: str, seed: int) -> dict:
    """paper headline ckpt = m1_8_stage3_train/ablation_retrain_*_stage3/best.pt
       (ablation_retrain_eval.py:99 pattern, aggregate.csv 'full' MAE=0.390 ≡ Table I 0.389)."""
    if model == "cg_mamba":
        p = _ROOT / f"runs/m1_8_stage3_train/ablation_retrain_full_s{seed}_stage3"
        cfg = json.load(open(_ROOT / f"runs/m1_7_train/ablation_retrain_full_s{seed}_stage2/config.json"))
        return {"kind": "cgm", "ckpt": p / "best.pt", "cfg": cfg,
                "hmm_dir": Path(str(HMM_TPL).format(seed=seed))}
    if model == "vanilla_mamba":
        p = _ROOT / f"runs/vanilla_mamba_final/d64_nl3_lr5e-04/seed{seed}"
        cfg = json.load(open(p / "results.json"))["config"]
        return {"kind": "vanilla", "ckpt": p / "vanilla_mamba_best.pt", "cfg": cfg}
    if model == "no_encgates":
        p = _ROOT / f"runs/m1_8_stage3_train/ablation_retrain_no_encgates_s{seed}_stage3"
        cfg = json.load(open(_ROOT / f"runs/m1_7_train/ablation_retrain_no_encgates_s{seed}_stage2/config.json"))
        return {"kind": "no_encgates", "ckpt": p / "best.pt", "cfg": cfg,
                "hmm_dir": Path(str(HMM_TPL).format(seed=seed))}
    if model == "no_env":
        p = _ROOT / f"runs/m1_8_stage3_train/ablation_retrain_no_env_s{seed}_stage3"
        cfg = json.load(open(_ROOT / f"runs/m1_7_train/ablation_retrain_no_env_s{seed}_stage2/config.json"))
        return {"kind": "no_env", "ckpt": p / "best.pt", "cfg": cfg,
                "hmm_dir": Path(str(HMM_TPL).format(seed=seed))}
    raise ValueError(model)


# ============================================================================
# HOOK 2 — model load (CGM/no_encgates: prepare_for_stage2 → load_state_dict)
# ============================================================================
def load_checkpoint(model: str, info: dict, device=DEVICE):
    if info["kind"] == "cgm":
        cfg = CGMambaConfig()
        net = CGForecaster(cfg)
        hmm = load_fitted_hmm(info["hmm_dir"])
        net.prepare_for_stage2(hmm)
        ck = torch.load(info["ckpt"], map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        net.load_state_dict(sd, strict=False)
        net.eval().to(device)
        return net, hmm
    if info["kind"] == "vanilla":
        c = info["cfg"]
        net = VanillaMambaForecaster(
            seq_len=c["seq_len"], pred_len=c["pred_len"], enc_in=c["enc_in"],
            d_model=c["d_model"], n_layers=c["n_layers"], d_state=c["d_state"],
            dt_rank=c["dt_rank"], expand=c["expand"], dropout=c.get("dropout", 0.0))
        ck = torch.load(info["ckpt"], map_location=device, weights_only=True)
        net.load_state_dict(ck)
        net.eval().to(device)
        return net, None
    if info["kind"] in ("no_encgates", "no_env"):
        from ablation_retrain import NoEncGatesCGForecaster, NoEnvCGForecaster
        klass = NoEncGatesCGForecaster if info["kind"] == "no_encgates" else NoEnvCGForecaster
        cfg = CGMambaConfig()                                            # ablation_retrain.py:243 패턴 (frozen 기본값 사용)
        net = klass(cfg)
        hmm = load_fitted_hmm(info["hmm_dir"])
        net.prepare_for_stage2(hmm)
        ck = torch.load(info["ckpt"], map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        net.load_state_dict(sd, strict=False)
        net.eval().to(device)
        return net, hmm
    raise ValueError(info["kind"])


# ============================================================================
# HOOK 3 — region inputs (split-aware: val | test_strict)
# ============================================================================
def _split_filter(split: str):
    if split == "test_strict":
        return "test", (lambda ep: ep >= TS_BOUNDARY)
    if split == "val":
        return "val", (lambda ep: True)
    raise ValueError(split)


def load_region_inputs(region: str, split: str, model_kind: str, cfg_dict: dict | None = None):
    """반환: (ds, ds_kind, epi_filter, region_df). region_df 는 origin/target ep 조회용."""
    df = _build_region_df(region)
    loader_split, epi_filter = _split_filter(split)
    if model_kind in ("cgm", "no_encgates", "no_env"):
        cfg = CGMambaConfig()
        ds = WeeklyDataset(df, split=loader_split, lookback=cfg.lookback,
                            horizon=max(cfg.horizons), norm=NORM)
        return ds, "dict", epi_filter, df
    c = cfg_dict
    ds = WeeklyMultiHorizonDataset(df, loader_split, NORM,
                                    lookback=c["seq_len"], pred_len=c["pred_len"])
    return ds, "tuple", epi_filter, df


# ============================================================================
# HOOK 4 — single-window forward; per-region iterator yields well-formed rows
# ============================================================================
def _forward_window(net, kind: str, batch, device=DEVICE):
    """단일 윈도우 → (mu_z[H], gamma_all[H,K]|None). mu_z 는 z-score 공간."""
    if kind == "vanilla":
        x, _y = batch
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = net(x)[0].cpu().numpy()
        return pred, None
    d = batch
    x = d["x"].unsqueeze(0).to(device)
    env = d["env"].unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            pred, inter = net(x, env, return_intermediates=True)
        except TypeError:
            pred = net(x, env)
            inter = None
    mu_z = pred[0].cpu().numpy()
    ga = inter["gamma_all"][0].cpu().numpy() if (inter is not None and "gamma_all" in inter) else None
    return mu_z, ga


def forward_region(net, hmm, kind: str, ds, ds_kind: str, region_df, epi_filter, cfg_max_h: int):
    """region 단위로 윈도우 전체 iterate.
       반환: 행 list — {origin_ep, target_ep, horizon, mu_z, y_true_raw, gamma_h[K]|None}"""
    eps_col = region_df["epiweek"].astype(int).to_numpy()
    wili_col = region_df["ili_weighted_pct"].astype(float).to_numpy()
    rows_out = []
    if ds_kind == "dict":
        # WeeklyDataset.windows: WindowSpec(start_row, end_row, target_row, target_epiweek)
        for i in range(len(ds)):
            w = ds.windows[i]
            origin_ep = int(eps_col[w.end_row])
            if not epi_filter(origin_ep):
                continue
            batch = ds[i]
            mu_z, ga = _forward_window(net, kind, batch)
            if np.isnan(mu_z).any():
                continue
            for h_idx, h in enumerate(HORIZONS):
                tgt_row = w.target_row - (cfg_max_h - h)
                rows_out.append(dict(
                    origin_ep=origin_ep,
                    target_ep=int(eps_col[tgt_row]),
                    horizon=h,
                    mu_z=float(mu_z[h_idx]),
                    y_true_raw=float(wili_col[tgt_row]),
                    gamma_h=(ga[h_idx].tolist() if ga is not None else None),
                ))
    else:                                                                 # tuple (vanilla)
        for i in range(len(ds)):
            end = int(ds.window_ends[i])
            origin_ep = int(eps_col[end])
            if not epi_filter(origin_ep):
                continue
            batch = ds[i]
            mu_z, _ = _forward_window(net, kind, batch)
            if np.isnan(mu_z).any():
                continue
            for h_idx, h in enumerate(HORIZONS):
                tgt_row = end + h
                rows_out.append(dict(
                    origin_ep=origin_ep,
                    target_ep=int(eps_col[tgt_row]),
                    horizon=h,
                    mu_z=float(mu_z[h_idx]),
                    y_true_raw=float(wili_col[tgt_row]),
                    gamma_h=None,
                ))
    return rows_out


# ============================================================================
# Forecast collection — per (model, seed, region, split) → rows
# ============================================================================
def collect_forecasts(model: str, seed: int, regions, splits) -> list[dict]:
    """모델 1개를 로드, 지정된 region/split 조합 모두 평가, row list 반환."""
    info = _resolve_checkpoint(model, seed)
    net, hmm = load_checkpoint(model, info)
    mu_k = hmm.means[:, 0] if hmm is not None else None
    s2_k = hmm.covars[:, 0, 0] if hmm is not None else None
    cfg_max_h = max(HORIZONS)
    all_rows = []
    for region in regions:
        for split in splits:
            ds, ds_kind, epi_filter, region_df = load_region_inputs(
                region, split, info["kind"], cfg_dict=info.get("cfg"))
            rows = forward_region(net, hmm, info["kind"], ds, ds_kind,
                                    region_df, epi_filter, cfg_max_h)
            if not rows:
                continue
            # APMD (CGM/no_encgates/no_env: gamma_h available)
            for r in rows:
                r["model"] = model
                r["seed"] = seed
                r["region"] = region
                r["split"] = split
                r["mu"] = r["mu_z"] * TARGET_STD + TARGET_MEAN
                r["y_true"] = r["y_true_raw"]                              # raw wILI (denorm)
                mu_z = r.pop("mu_z"); _y = r.pop("y_true_raw")
                if r["gamma_h"] is not None and mu_k is not None:
                    g = np.array(r["gamma_h"])                             # [K]
                    mu_hmm_z = float((g * mu_k).sum())
                    sw_z = float((g * s2_k).sum())
                    sb_z = float((g * (mu_k - mu_hmm_z) ** 2).sum())
                    st_z = max(sw_z + sb_z, 1e-12)
                    r["s2_within"] = sw_z * (TARGET_STD ** 2)
                    r["s2_between"] = sb_z * (TARGET_STD ** 2)
                    r["s2_total"] = st_z * (TARGET_STD ** 2)
                    r["sigma"] = float(np.sqrt(r["s2_total"]))
                else:
                    r["s2_within"] = r["s2_between"] = r["s2_total"] = np.nan
                    r["sigma"] = np.nan
                r.pop("gamma_h")
            all_rows.extend(rows)
    return all_rows


# ============================================================================
# Driver 1 — full per-origin dump
# ============================================================================
def dump_forecasts(models=None, regions=None, splits=None, out_name="per_origin_forecasts.parquet"):
    models  = models  or MODELS
    regions = regions or REGIONS
    splits  = splits  or SPLITS
    rows = []
    for model in models:
        for seed in SEEDS:
            r = collect_forecasts(model, seed, regions, splits)
            rows.extend(r)
            print(f"[dump] {model:14s} seed{seed}: +{len(r):5d} rows  (total {len(rows):,})", flush=True)
    df = pd.DataFrame(rows)
    path = OUT_DIR / out_name
    df.to_parquet(path, index=False)
    print(f"[dump] wrote {len(df):,} rows → {path}")
    print(df.groupby(["model", "split"]).size())
    return df


# ============================================================================
# Driver 2 — per-region smoothed gamma  (field 6)
# ============================================================================
def smooth_gamma_per_region(seed: int = 42):
    """frozen HMM (seed=42, K-selection seed-invariant 가정) + featurize_raw 경로 (G1 검증).
       rollout γ_all 아님 — full-sequence smoothed posterior γ_t."""
    from scripts.m1_4_phase_dynamics_search import featurize_raw, RAW_COLS_V3
    info = _resolve_checkpoint("cg_mamba", seed)
    net, _ = load_checkpoint("cg_mamba", info)
    pm = net.phase_module
    for region in REGIONS:
        df_full = _build_region_df(region)
        # Δx lag 으로 augment 후 1주 손실 — 시간 정렬 위해 [1:] epiweek 으로 키 매핑
        x_raw_np = featurize_raw(df_full, NORM, RAW_COLS_V3)                          # [T, 3]
        x_raw_t = torch.tensor(x_raw_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            x_aug = pm._augment_features(x_raw_t)                                       # [1, T-1, 6]
            gamma = pm._torch_forward_backward(x_aug)[0].cpu().numpy()                  # [T-1, K]
        assert np.allclose(gamma.sum(1), 1.0, atol=1e-5), f"{region}: posterior 합≠1"
        eps = df_full["epiweek"].astype(int).to_numpy()[1:]                             # augment Δx lag 정렬
        np.savez(OUT_DIR / f"gamma_smoothed_{region}.npz", gamma=gamma, target_ep=eps)
        print(f"[gamma] {region}: shape={gamma.shape}  eps=[{eps[0]}..{eps[-1]}]")


# ============================================================================
# build_load_data — regime_shift_experiment.py 의 load_data() 대체
# ============================================================================
def build_load_data():
    df = pd.read_parquet(OUT_DIR / "per_origin_forecasts.parquet")
    name = {"cg_mamba": "CGM", "vanilla_mamba": "VANILLA",
            "no_encgates": "NO_ENCGATES", "no_env": "NO_ENV"}
    df["M"] = df["model"].map(name)

    test = df[df.split == "test_strict"]
    val  = df[df.split == "val"]

    # 5-seed 평균 점예측 — KEY = (target_ep, h). experiment 의 t = target_ep 규약.
    pt = (test.groupby(["M", "region", "target_ep", "horizon"])["mu"].mean().reset_index())
    point_forecast = {m: {} for m in name.values()}
    origins_map = {r: sorted(test[test.region == r].target_ep.unique().tolist()) for r in REGIONS}
    for _, r in pt.iterrows():
        point_forecast[r.M].setdefault(r.region, {})[(int(r.target_ep), int(r.horizon))] = r.mu

    yt_lkp = test.groupby(["region", "target_ep", "horizon"])["y_true"].first()
    def y_at(region, t, h):  return float(yt_lkp[(region, t, h)])

    # val 잔차 (national, pooled across seeds, per horizon)
    valn = val[val.region == "national"]
    val_resid = {m: {} for m in name.values()}
    for (m, h), g in valn.groupby(["M", "horizon"]):
        val_resid[m][int(h)] = (g["y_true"] - g["mu"]).to_numpy()

    # APMD (CGM) — same target_ep keying
    cgm = test[(test.M == "CGM") & test.s2_within.notna()]
    apmd = {r: {} for r in REGIONS}
    agg = cgm.groupby(["region", "target_ep", "horizon"]).agg(
        sw=("s2_within", "mean"), sb=("s2_between", "mean"), st=("s2_total", "mean")).reset_index()
    for _, r in agg.iterrows():
        apmd[r.region][(int(r.target_ep), int(r.horizon))] = (r.sw, r.sb, r.st)

    # observed wILI + smoothed gamma + sd_train
    observed = {r: _load_observed_wILI(r) for r in REGIONS}
    def gamma_fn(region):
        d = np.load(OUT_DIR / f"gamma_smoothed_{region}.npz")
        return d["gamma"], d["target_ep"]
    sd_train = _train_wILI_std()

    return dict(point_forecast=point_forecast, observed_wILI=observed,
                origins_fn=lambda r: origins_map[r], y_at_fn=y_at,
                national_val_residuals=val_resid, gamma_smoothed_fn=gamma_fn,
                sd_train=sd_train, apmd_components=apmd)


# ============================================================================
# HOOK 7+8 — observed wILI + train std
# ============================================================================
def _load_observed_wILI(region: str):
    """γ 타임라인(eps 200141..202535, len 1228) 에 정렬된 wILI np.array 반환.
       experiment 의 `assert len(obs)==len(γ)` 통과용."""
    df = _build_region_df(region)
    ep2val = dict(zip(df["epiweek"].astype(int), df["ili_weighted_pct"].astype(float)))
    g = np.load(OUT_DIR / f"gamma_smoothed_{region}.npz")
    eps = g["target_ep"]
    arr = np.array([ep2val.get(int(e), np.nan) for e in eps], dtype=float)
    if np.isnan(arr).any():
        n_miss = int(np.isnan(arr).sum())
        raise ValueError(f"[{region}] observed wILI 빠진 epiweek {n_miss}개 (γ timeline 정렬 실패)")
    return arr

def _train_wILI_std() -> float:
    df = pd.read_csv(SPLIT_CSV)
    return float(df.loc[df.split == "train", "ili_weighted_pct"].std())


# ============================================================================
# Preflight (3 reproduction gates, ~5 min)
# ============================================================================
def preflight(include_no_env: bool = True):
    """National only, test_strict, all models 5-seed.
       Gate G2/G3: per-model 5-seed avg MAE 를 PAPER_REF 와 비교."""
    models = MODELS_PLUS_NOENV if include_no_env else MODELS
    print(f"[preflight] models={models}  regions=[national]  split=test_strict")
    all_rows = []
    for model in models:
        for seed in SEEDS:
            r = collect_forecasts(model, seed, ["national"], ["test_strict"])
            all_rows.extend(r)
            mae = float(np.mean([abs(x["mu"] - x["y_true"]) for x in r]))
            print(f"  {model:14s} s{seed}: n={len(r):4d}  MAE={mae:.4f}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_DIR / "preflight_national.parquet", index=False)
    print(f"\n[preflight] === 5-seed avg national test_strict MAE vs paper ===")
    print(f"{'model':<16s} {'seed-mean MAE':>14s} {'paper ref':>11s} {'Δ':>9s}  status")
    ok_all = True
    for model in models:
        sub = df[df.model == model]
        # per-seed MAE then 5-seed mean
        per_seed = sub.groupby("seed").apply(
            lambda g: (g.mu - g.y_true).abs().mean()).to_numpy()
        mae5 = float(per_seed.mean())
        ref = PAPER_REF_MAE_AVG.get(model)
        if ref is None:
            print(f"{model:<16s} {mae5:>14.4f} {'  N/A':>11s} {'-':>9s}  (vanilla: no ref check)")
            continue
        d = mae5 - ref
        ok = abs(d) < 0.02                                                 # ±0.02 tol
        ok_all &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"{model:<16s} {mae5:>14.4f} {ref:>11.4f} {d:>+9.4f}  {flag}")
    print(f"\n[preflight] {'ALL GATES PASS' if ok_all else 'SOME GATES FAIL'}")
    return ok_all


# ============================================================================
# Gamma check (G1: reproduce stored gamma.npy from national feature pipeline)
# ============================================================================
def gamma_check(seed: int = 42):
    """저장된 gamma.npy(834,3) 와 driver가 재구성한 γ 를 비교.
       transform/feature 순서가 학습 때와 동일한지 구성적 증명."""
    from scripts.m1_4_phase_dynamics_search import featurize_raw, RAW_COLS_V3
    ref_path = _ROOT / f"runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed{seed}/gamma.npy"
    gamma_ref = np.load(ref_path)                                          # [834, 3]
    print(f"[gamma_check] ref gamma shape={gamma_ref.shape}, row-sum mean={gamma_ref.sum(1).mean():.6f}")

    # 학습 구간 재구성: TRAIN_START_EPIWEEK=200240 + split=='train' (HMM 학습 셋과 정확 일치)
    df = pd.read_csv(SPLIT_CSV)
    df = df[(df["split"] == "train") & (df["epiweek"] >= 200240)].reset_index(drop=True)
    print(f"[gamma_check] train seg2 rows = {len(df)}  (expected 835)")
    x_raw_np = featurize_raw(df, NORM, RAW_COLS_V3)                        # [L, V_raw=3]

    # CGM seed42 모델 로드(HMM cached)
    info = _resolve_checkpoint("cg_mamba", seed)
    net, hmm = load_checkpoint("cg_mamba", info)
    pm = net.phase_module
    x_raw_t = torch.tensor(x_raw_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, L, 3]
    with torch.no_grad():
        x_aug = pm._augment_features(x_raw_t)                              # [1, L-1, V_aug=6]
        gamma_new = pm._torch_forward_backward(x_aug)[0].cpu().numpy()     # [L-1, K]
    print(f"[gamma_check] new gamma shape={gamma_new.shape}")

    # 겹치는 구간 정렬 (둘 다 _augment_features 적용 후 길이; 같아야 정상)
    T = min(gamma_ref.shape[0], gamma_new.shape[0])
    diff = np.abs(gamma_ref[:T] - gamma_new[:T])
    print(f"[gamma_check] |Δγ| over T={T}: max={diff.max():.2e}  mean={diff.mean():.2e}")
    # float32(torch) vs float64(numpy EM) 차이로 max~5e-4 발생; mean~1e-5 면 사실상 동일.
    ok = (diff.max() < 1e-3) and (diff.mean() < 1e-4)
    print(f"[gamma_check] {'PASS — feature transform/순서/state ordering 일치' if ok else 'FAIL'}")
    if not ok:
        print(f"  ref[0]={gamma_ref[0]}\n  new[0]={gamma_new[0]}")
        print(f"  ref[-1]={gamma_ref[-1]}\n  new[-1]={gamma_new[-1]}")
    return ok


# ============================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if   cmd == "preflight":  preflight()
    elif cmd == "gamma_check": gamma_check(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 42)
    elif cmd == "dump":  dump_forecasts(models=MODELS_PLUS_NOENV)
    elif cmd == "gamma": smooth_gamma_per_region(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 42)
    else:
        print("usage: python regime_shift_drivers.py [preflight|gamma_check [seed]|dump|gamma [seed]]\n"
              "  preflight    → national 5-seed test_strict MAE vs paper (G2/G3)\n"
              "  gamma_check  → 저장된 gamma.npy 재현 (G1)\n"
              "  dump         → 풀 per_origin_forecasts.parquet (G1~G3 통과 후)\n"
              "  gamma        → per-region smoothed γ npz (field 6)")
