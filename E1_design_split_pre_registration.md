# E1 (design-split HPO) 사전등록 commit — γ (v2, 5 정정 반영)

> **위상**: PC2-a PASS 직후, E1 launch 전 *integrity gate*.
>
> **목적**: reviewer #4 (design-time exposure) 의 *모든 채널* (architecture HPO + HMM fit + scaler + s_h) 을 닫는다. 결과 보고 수정 금지.
>
> **정직 hedge** (over-claim 차단): E1 은 #4 의 *누설 채널* 을 닫지만 n=1 시즌 (FluSight 2018-19) 는 그대로. FluSight 순위 주장은 E1 후에도 "single-season hindcast" 로 hedge 유지. "E1 이 #4 완전 해결" 로 적지 말 것.

---

## γ.1 — SPLIT CUT (frozen)

| split | epiweek | 시즌 | rows | 용도 |
|---|---|---|---:|---|
| design-train | 200140 – 201539 | 14 (W40-2001 ~ W39-2015) | ~714 | HPO config 학습 + **HMM/scaler 재적합 원천** |
| **design-val** | **201540 – 201839** | **3 (W40-2015 ~ W39-2018)** | **~153** | HPO config 선택 |
| (final-train) | 200140 – 201839 | 17 (원 train) | 868 | E1 winner config 으로 *최종 모델* 학습 |
| (held-out) | 201840 – 202010 | FluSight 2018-19 포함 | 75 | E1 *이후* 평가 only |
| (test_strict) | 202240 – 202535 | 3 | 152 | 추가 평가 (regional transfer) |

**핵심 차단**: design-val 의 마지막 = W39-2018 = FluSight 평가시즌 시작 *직전 주*. HPO selection 무접촉.

**design-val 3 시즌 유지 근거**: rows 자체는 153 (marginal) 이지만 Cov95 안정성을 **5-seed pooling** 으로 해결 (γ.3 참조). 시즌 수를 줄여 design-train 늘려도 아키텍처 선택 robustness 는 ±3 시즌에 둔감.

**design-train 정의 차이 (재현성 노트)**: design-train = epiweek ≤ 201539. 단 사용 컴포넌트마다 시작점 다름:
- **HMM 재적합 (γ.4)**: 200240 – 201539 (seg2-only, 679 rows). 2002-W21~W39 데이터 갭으로 EM 시퀀스 연속성 위해 seg2-only convention 적용 (원 m1_4 패턴 honor)
- **Scaler 재적합 (γ.5)**: 200140 – 201539 (전체, 712 rows). 단순 mean/std 통계는 gap 무관, 원 scaler convention (full train range) honor
- 둘 다 design-val (≥201540) 은 *완전 제외* → leak-free. 시작점 차이는 *각 컴포넌트의 원 convention 보존* 이지 누설 채널 아님

---

## γ.2 — HPO GRID (κ 재검증 결과: **45 runs 확정**)

**선택 차원** (architectural):
| dim | winner | grid | 누설 채널? | 근거 |
|---|---:|---|---|---|
| `K_phase` | 3 | **{3} 고정** | *eval-WIS 가 아니라 train 재현성(κ)* | §IV-7 K-ablation κ=1.0 + γ.4 재검증 κ_min=0.9459 ≥ 0.8 → 안정 |
| `n_layers` | 3 | **{2, 3, 4}** | YES (eval 통계 영향 가능) | winner ±1 |
| `d_model` | 64 | **{32, 64, 128}** | YES | winner ±1 |

**grid size 확정** (κ_min=0.9459 ≥ 0.8 통과, 2026-06-13 측정):
- **K=3 고정 → grid = 3 × 3 = 9 configs × 5 seeds = 45 runs** ✓

**고정 (HPO 변동 X — paper CG_TOP1_HP 9개 HP override, ablation_retrain.py:65 일치)**:
- `CG_TOP1_HP = {"gate_lr": 1e-3, "backbone_lr": 1e-4, "lookback": 104, "hmm_lr_ratio": 0.01, "state_embed_lr_ratio": 0.01, "env_lr_ratio": 0.001}`
- `OTHER_LR_BASE = 1e-4` (Stage 3)
- Stage 2: 200 epochs, batch 32, LR/wd/clip 위 dict
- **Stage 3: 30 epochs, patience=10** (paper m1_9_hpo_phase2 protocol)
- HMM seed=42, hmm_n_init=5, hmm_reg_covar=5e-3 (γ.4 재적합과 동일)

**🔴 Bug fix 기록 (2026-06-14)**: 이전 코드 `dataclasses.replace(CGMambaConfig(), seed, n_layers, d_model, data_csv, norm_json)` 는 CGMambaConfig() default (lookback=156, backbone_lr=5e-5) 사용 → paper baseline 과 다른 HP env. ablation_retrain.build_frozen_hpo_cfg 패턴 미적용. 정정 후 9개 HP 모두 명시 override. Buggy 결과 `runs/_archive_buggy_HP_2026-06-14/` 로 archive + SUPERSEDED 마킹.

---

## γ.3 — SELECTION CRITERION (정정 LOCKED — paper m1_9_hpo_phase2 일치)

**🔴 정정 (2026-06-14, bug fix — 결과 본 후 reframe 아님)**:

이전 spec ("argmin WIS s.t. Cov95 ∈ [0.90, 0.96], fallback [0.85, 0.99]") 는 **paper 의 selection 기준을 모르고 정한 spec**. paper m1_9_hpo_phase2.py:268 *"Selection: val_total (= stage3_best_val) ascending"* 확인 후 정정.

→ E1 의 목적 (paper 의 *누설만* 격리) 위해 **selection criterion 도 paper 와 동일** 해야 confound 제거.

**정정 기준** (LOCKED):
```
config* = argmin_{c}  mean_over_seeds( stage3_best_val_total_c )
```

- `val_total` = MSE + 0.3·MASE ([src/utils/losses.py:158](src/utils/losses.py#L158) `cg_mamba_loss`, λ_mase=0.3)
- *순수 점예측 loss* — calibration / Cov95 / WIS 무관
- 5 seed mean ascending. ties = first-listed (deterministic)
- band / fallback ladder **폐기**

**Diagnostic only (selection 무관)**: 각 config 의 pooled Cov95 / WIS / MAE 보고 (paper 와 동일 protocol 위 측정). § IV-D 표에 함께 명시.

**금지**: 사후 새 metric / band 재도입 / val_total 외 score selection.

---

## γ.4 — HMM 재적합 (완료, 누설 채널 #2 차단)

**현재 누설**: `runs/m1_4_phase_dynamics_main/V_raw3_regcov5e-03_K3_seed42/` 의 HMM 은 **train 17 시즌 (200140-201839) 위 EM 적합** → design-val (201540-201839) 통계가 HMM 의 means/covars/transition 에 누설.

**조치 (실행 완료, 2026-06-13)**:
1. **HMM design-train 재적합**: m1_4 multi-init EM 패턴으로 *design-train seg2 (200240-201539) only* 재실행. seed ∈ {42, 123, 456}, n_init=5, K=3, reg_covar=5e-3. 출력: `runs/m1_4_design_split/V_raw3_regcov5e-03_K3_seed{42,123,456}/`
2. **K=3 κ 재검증 (완료)**: cross-seed Cohen's κ 측정
   - κ_min = **0.9459** ≥ 0.8 → K=3 고정 PASS → γ.2 grid = **45 runs 확정**
   - μ_k (z-ili) 3-seed 일관 (low/neutral/peak phase identity preserved)
   - 해석 (사후 변경 금지): "K=3 가 design-train(seg2 678 pts) 위에서도 안정. 데이터 크기 -3 시즌 변화에 robust"
3. **Sanity ② — design-val state collapse 감지 ⚠**:
   - 모든 seed 에서 *extreme-low (off-season) state* 의 design-val 점유율 **0.6%** (~1 row / 155)
   - 원인: design-val (2015-18) 3 시즌 자체가 *우연히 off-season trough 결여* — *시간적 누락*이지 *구조적 collapse* 아님 (PC0 의 region collapse 와 다름)
   - 영향: HPO selection 이 off-season/low-phase 에 *blind*. 특정 실패 모드 — over-wide config 가 peak/neutral 만으로 calibration band 통과 가능. 심각도 modest (빠진 게 *easy* off-season, 어려운 peak/transition 은 cover)
   - 조치: split 재설계 X (locked γ 재개봉 과함). γ.6 에 off-season guard 1줄 추가 (아래), §V-D + response letter 에 *능동 disclosure* (γ.7)
4. **HPO 의 각 config 학습 시** 이 HMM (design-train fit) 사용. train 17 시즌 HMM 재사용 금지.

---

## γ.5 — Scaler / s_h / env encoder 재적합 (완료, 누설 채널 #3 + #4 차단)

**현재 누설**: `data/processed/normalization_params.json` 의 mean/std 가 **train 17 시즌 위 적합** → design-val 통계가 z-score 파라미터로 누설.

**조치 (실행 완료, 2026-06-13)**:
1. **Scaler design-train 재적합 ✓**: scripts/refit_scaler_design_train.py 로 design-train (200140-201539, split=='train', 712 rows) 만으로 재계산.
   - convention 검증 (full train 재계산 → 원 JSON bit-level 일치) PASS, ddof=0 (population std) 확인
   - 출력: `data/processed/normalization_params_design_train.json`
   - 누설 보정 deltas (vs 원 17 시즌): ili_mean −0.046, ili_std −0.052, temp_mean −0.21, hum_mean −0.11 (작지만 non-zero — 누설 실재)
2. **HPO 의 각 config 학습 시** 이 scaler 사용. 17 시즌 scaler 재사용 금지.
3. **env encoder design-train 재pretrain ✓** (누설 채널 #4):
   - 원 `runs/m1_7_env_pretrain/env_encoder.pt` 는 *full train (200140-201839)* env data 위 pretrain → design-val (201540-201839) 통계 누설 (env representation 이 FluSight era 의 seasonal pattern 학습)
   - **EnvModule.encoder = `Linear(V=2, H=32) → ReLU → Linear(H=32, D=d_model)`** — 두 번째 layer 가 d_model 의존 → d_model 별 3 pretrain 필요
   - 조치: `scripts/refit_env_encoder_design_train.py` 로 design-cut CSV + design-train norm 사용해 d_model ∈ {32, 64, 128} 각각 100 epochs pretrain → `runs/m1_7_env_pretrain_design/env_encoder_d{32,64,128}.pt`
   - 검증: 모두 PLAN §5.1 sanity PASS (val_mse < random × 0.5). val_mse: d32=0.0078, d64=0.0082, d128=0.0055
   - HPO 의 각 (n_layers, d_model, seed) run 이 d_model 별 env ckpt 사용 (e1_hpo.py `DESIGN_ENV_CKPT_TPL` dispatch)

4. **selection 분산 = raw design-train HMM emission 분산 단독 (HARD LOCK, s_h 사용 금지)**:
   - HPO selection 의 Cov95/WIS 계산은 **design-train HMM 의 raw √s²_total** (within + between, post-hoc scaling 없음) 만 사용
   - **s_h 옵션 완전 삭제** — selection 단계에 어떤 s_h (national val fit / design-train fit / leave-one-out) 도 사용 금지
   - 근거 1 (누설 차단): s_h 가 끼면 fit 데이터의 통계가 selection 에 샘. national val (2018-20) fit s_h 는 held-out era 통계 직접 누설. design-train fit s_h 도 cross-fold complexity. raw 면 design-train 만 의존 → 완전 leak-free
   - 근거 2 (제약 의미 보존): s_h 로 calibrate 하면 모든 config 가 nominal 로 맞춰져 calibration band [0.90, 0.96] non-binding → selection 이 val-WIS 단독 으로 퇴화 (γ.3 가 막으려던 그것). raw 면 band 가 config 의 *내재적* calibration 을 변별 → floor 의 "아키텍처가 내재적으로 calibrated" 주장과 정합
   - 근거 3 (가능성 확인): PC2-a 의 phase-anchored raw Cov95 = 0.955 (regional test_strict 위) → raw 가 nominal band 안에 들어옴이 *현존 데이터로* 확인. design-val 도 도달 가능성 충분
   - band 못 맞히면 → fallback ladder (γ.3) → FAIL 은 정직한 결과 ("내재적으로 calibrated 한 config 없음" = floor-full-negative)
   - **s_h 는 E3 (학습 head 대안) 또는 배포 calibration 의 영역 only**. selection 무관

---

## γ.6 — Stage 2 + Stage 3 protocol (정정 LOCKED) + 실행 후 보고

**🔴 Bug 2 fix (2026-06-14)**: 이전 e1_final_train.py 는 Stage 2 만 학습 → m1_7_train/.../best.pt 사용. paper headline ckpt = `runs/m1_8_stage3_train/.../best.pt` (Stage 2 + Stage 3). 정정:
- **e1_hpo (selection)**: Stage 2 (200 ep) + Stage 3 (30 ep, **patience=10**, paper `m1_9_hpo_phase2` mirror) → Stage 3 best.pt 위 design-val 평가
- **e1_final (winner 학습)**: Stage 2 (200 ep) + Stage 3 (30 ep, **patience=0** — full 30ep no early stop, paper `ablation_retrain.py:384` mirror = *Table I headline ckpt 출처*) → Stage 3 best.pt 가 paper headline 비교 대상
- **CATCH A (2026-06-14)**: paper 의 두 Stage 3 protocol 이 다름 — selection (m1_9, patience=10) vs final-train (ablation_retrain, patience=0). E1 도 그대로 분리 mirror.
- e1_final_eval: ckpt path = `runs/m1_8_stage3_train/e1_final_{config}_s{seed}_stage3/best.pt`
- paper m1_9_hpo_phase2 의 selection = *stage3_best_val* 위 ranking (γ.3 정정 일치).

**🟡 CATCH B (2026-06-14, 조건부 verify)**: `runs/m1_7_env_pretrain_final/` 에 d64 + d128 만 존재. e1_hpo 의 winner 가 *d32* 이면 (가능성 작지만 0 아님) — final-train env d32 추가 pretrain 필요 (`scripts/refit_env_encoder_final.py` 에 D_MODELS=[32] 추가, ~5분). e1_final_train 시작 *전* verify step.

### 실행 후 보고 (frozen)

- HPO 전체 결과 (45 configs × Cov95_pooled, WIS_pooled) → appendix table 로 *전부* 공개
- winner config (K/depth/dim) vs 현재 winner (3/3/64) 비교 명시
- band relaxation event 명시 (있으면)
- HMM design-train κ 값 (= 0.9459 측정 완료) + grid 분기 결정 (= 45) 보고
- 최종 모델 (winner config × 5 seeds × full train 200140-201839, design-train HMM 사용 X — final-train scaler 와 final-train HMM 별도) 의 held-out (201840-202010) Cov95/WIS

**※ 주의**: 최종 모델 학습 시점에는 *원 train 17 시즌 HMM + scaler* 사용 (final-train 데이터 자체에 fit). E1 HPO 의 design-train HMM/scaler 와는 다름. 이건 누설이 아님 — held-out 평가 데이터(val period) 와 분리.

**📌 OFF-SEASON GUARD (γ.4 의 design-val collapse 보완, pre-register)**:
- E1 winner config 의 held-out (201840-202010, FluSight 2018-19 full-cycle 포함) 평가 시 **off-season / low-phase 구간을 별도로 떼서 Cov95 확인** 필수.
- 정의: held-out epiweek 중 ili_weighted_pct < (mean - 1σ) 또는 design-train HMM 으로 Viterbi 했을 때 *extreme-low state* 점유 구간
- *심하게 over-cover* (Cov95 > 0.99) → flag, 본문 명시
- 근거: design-val (2015-18) 이 우연히 off-season trough 결여 → HPO selection 이 low-phase calibration 에 blind 했음. held-out 의 off-season 구간을 별도 확인해서 selection 의 blindness 가 *deployment time* 에 실패로 이어지지 않았는지 검증.
- 비용: 새 실험 X (held-out 평가는 어차피 γ.6 에 있음), off-season subset slicing 1 단계 추가만.

---

## γ.7 — Commit 규약 + Disclosure (락 §5 honor)

**Commit 규약**:
- γ.1 ~ γ.6 의 *수치/규정 그대로*, 실행 후 수정 금지
- band relaxation 은 frozen ladder (1→2→fail) 안에서만, ad-hoc 새 band 금지
- E1 FAIL → floor-full-negative 분기, "phase 가 작동 (PC2-a)" 주장도 *주변 strength* 강등
- v7 reframe 금지

**🔴 Bug fix disclosure (2026-06-14, *결과 본 후 reframe 아님* — 코드 bug 정정)**:

- Bug 1 (HP override 누락): e1_hpo + e1_final build_cfg 가 CGMambaConfig() default 사용. paper CG_TOP1_HP 9개 HP override 누락. → 정정 (ablation_retrain.build_frozen_hpo_cfg 패턴 적용).
- Bug 2 (Stage 3 누락): e1_final 가 Stage 2 만 학습. paper headline = Stage 2 + Stage 3 ckpt. → 정정.
- γ.3 정정: selection criterion calibration-제약 WIS → val_total ascending (paper m1_9_hpo_phase2 일치).
- 비대칭 HP 환경 (CGM=clean, baseline=val-HPO leaky) §V-D 명시 disclosure 필요: *"E1 의 fairness 목표는 paper HP 환경과 일치 + 누설 채널 4종 차단. 아키텍처(n_layers, d_model)만 clean 재선택, 나머지 HP는 paper 환경 유지"*.

**Disclosure 메모 (§V-D + response letter 에 *능동 작성* — 묻기 전에 밝힘)**:

0. **누설 채널 4종 closure 정확 enumeration**:
   - (1) selection-split (γ.1) — design-val 이 평가 시즌과 시간 분리
   - (2) HMM-refit (γ.4) — design-train 위 EM 재적합, κ_min=0.9459 PASS
   - (3) scaler-refit (γ.5) — design-train 712 rows 위 z-score 재계산
   - (4) env-pretrain-refit (γ.5) — d_model 별 design-train env data 위 재pretrain
   - **체계적 audit (Audit 1–4) 로 5번째 채널 negative 확정**: 다른 pretrained ckpt 없음, scaler 외 global stat 없음, HMM hyperparam (K=3 + reg_covar=5e-3) 은 *full-train reproducibility/numerical robustness* 선택이라 45 config 에 상수 → selection 무편향, feature engineering (augment Δx, log1p) 은 sample-내부 연산 만
   - 문구: *"4 leak channels closed + systematic audit found no 5th"*
1. **n=1 시즌 hindcast hedge 영구**: E1 이 *4 누설 채널* 은 닫지만 *single-season hindcast* (FluSight 2018-19) 는 그대로. 본문/letter 어디에도 "E1 이 reviewer #4 완전 해결" 로 적지 말 것. *"4 누설 채널 차단 + n=1 시즌 한계 잔존"* 으로.
2. **design-val off-season 결여 한계 (γ.4 catch)**: *"design-val (W40-2015 ~ W39-2018) 3 시즌이 우연히 off-season trough 결여 → HPO selection 이 low-phase calibration 에 blind. held-out off-season 별도 확인 (γ.6) 으로 보완. peak/transition 같은 hard regime 은 cover 되므로 *modest* limitation, 그러나 active disclosure"*
3. **PC0 의 phase=anchor 가설 사망**: §V-D 에 PC0 결과 (state identity collapse → floor commit) 명시. *"phase-emission 의 zero-shot regional transfer 시 K=3 의 한 state 가 전 region 에서 ~0% 점유로 collapse — national 학습 phase 정의가 regional 에 부분만 transfer"*. 이게 paper 의 *negative-result content*.
4. **PC2-a 의 transfer-한정 mechanism (PASS, 단 strength 한정)**: §IV-x_region 또는 §V-D 의 mechanism note 로만. headline 부활 금지. *"transfer regime 에서 phase-anchored 분산이 constant scalar 대비 Cov95 우위 (Δ +0.217, CI 0 제외, 10/10 region). WIS 동급 — calibration parity + sharpness 양보 framing 과 정합"*. **E1 winner 변경 후 (n3_d64 → n4_d128) PC2-a 재확인 필수** — 새 모델에서도 phase-anchored > s_h 가 유지되는지 cheap 검증.

5. **α.3 transparent dual reporting (raw + E3-calibrated, reviewer #2 응답 전환)**:
   - E1 결과로 *raw HMM 분산 = over-cover (~0.98)* 가 모든 9 configs 에서 확인 → paper 의 *post-hoc s_h tuned 0.889* 가 *아키텍처 내재 calibration 이 아니라 tuning 인공물* 임이 데이터로 드러남 (reviewer #2 확증)
   - 응답 전환 (회피가 아니라 정면): held-out + test_strict 각 영역에서 **raw Cov95 (over-cover ~0.98) + E3-calibrated Cov95 (~nominal 0.95) 둘 다 나란히 보고**
   - 어휘: *"raw 분산은 intrinsic over-conservative → grid-search s_h (post-hoc) 가 아닌 *학습된 end-to-end head* 로 보정 → ~nominal"* — reviewer #2 의 "eq.13 s_h = conformal 과 다를 바 없다" 비판을 *"post-hoc 을 learned 로 교체했다"* 로 정면 응답
   - 헤드라인 어휘 정정: 기존 *"SARIMA-class intrinsic calibration"* → *"learned end-to-end calibration + intrinsic conservativeness 정직 보고"*

6. **E3 head calibration-set lock (LOCKED 사전등록, 결과 본 후 변경 금지)**:
   - **원칙 3개**:
     (i) **in-sample 절대 금지** — eval 데이터에 head fit 후 그 데이터 coverage 보고 = 순환
     (ii) **prospective**: cal-set 은 eval 기간보다 *시간상 이전* (배치 시나리오 그대로)
     (iii) **FluSight period 침해 금지**: FluSight 2018-19 시즌을 cal-set 으로 쓰면 그 eval 더러워짐
   - **매핑 (LOCKED)**:
     - **test_strict (W40-2022 ~ W35-2025) eval**: head WIS-loss fit on **held-out (W40-2018 ~ W10-2020)** — prospective, model 학습 데이터 외 (held-out은 원 split val), FluSight 평가는 raw-only 라 침해 X
     - **FluSight 2018-19 eval**: **raw-only 보고** (head 적용 안 함). 순위는 *점예측 (mu)* 기반이라 head=σ² 변경 무관. 추가로 *raw over-cover (~0.98) 직접 disclosure* 가 reviewer #2 의 "post-hoc s_h 가 인공물" 비판에 정면 응답
     - **PC2-a 재확인 (regional test_strict, n4_d128)**: raw HMM 분산 만 사용 (γ.5 narrative lock 그대로 — PC2-a 는 *selection level mechanism test* 이지 *applied calibration test* 아님)
   - 사전등록 어휘: *"E3 head 는 평가 기간보다 시간상 이전인 cal-set 에 WIS-loss 로 fit (prospective recalibration); test_strict 는 2018-2020 위 fit, FluSight 는 raw-only. eval 데이터에 절대 fit 안 함. 결과 본 후 cal-set 변경 금지."*

7. **Compact 헤드라인 강등 + efficiency-alternative angle 보존**:
   - E1 winner = **n4_d128 (506K = LSTM 의 63.5%)** ≠ 기존 paper n3_d64 (117K = 14.7%) → "*가장 작은 DL*" headline 깨짐
   - 정직 reposition: *headline model = n4_d128*, *efficiency alternative = n3_d64 (14% WIS 비용으로 4.3× param 절약, diminishing returns)*
   - Compact 절대 headline 강등, *trade-off curve* 로 표현 (단일 점 자랑 → 곡선 정직)

---

## 다음 행동 (정렬, 상태)

```
[1] κ 재검증 ✓ (κ_min=0.9459 ≥ 0.8 → grid 45 확정, design-val collapse caveat 박힘)
[2] Scaler 재적합 ✓ (normalization_params_design_train.json 저장, 보정 deltas 측정)
[3] γ lock — *현재 위치*, 사용자 승인 대기
[4] E1 HPO launch (45 runs × 5 seeds = 225 학습, ~수 시간 GPU)
[5] E1 도는 동안 병행: α (cluster_bootstrap util) + β (hmm_interval 정리) + E3 head 코드
[6] E1 완료 → winner 확정 → PC2-a 재확인 (parquet 재실행, 쌈) + E3 (Cov95≥0.92 threshold) + E2/E4 적용
[7] §2 reposition + §3 양보 + response letter (γ.7 disclosure 메모 반영)
```
