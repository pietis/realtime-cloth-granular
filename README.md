# Real-time Coupling of Discrete Grains and Porous Cloth Manifolds with Hysteretic Mass Transfer

> Classical Mechanics I (2026-1) 학기 프로젝트
> *A JKR-informed cross-solver mass-transfer operator between MPM granular flow and XPBD cloth surface reservoir.*

## 한 줄 요약

JKR(Johnson-Kendall-Roberts) 부착 에너지를 **XPBD constraint projection의 phase-transition criterion**으로 재정의해서, 모래(MPM) ↔ 천(XPBD) 사이의 **mass + linear momentum 보존적 가역 전이**를 게임 framerate에서 작동시키는 cross-solver coupling operator.

## 무엇이 새로운가

다음 5개가 **동시에** 만족되는 보고된 시스템 0건 (3-agent adversarial verification 통과 — Codex + Gemini + Claude 토론):

1. **Real-time** (≥30 fps, single GPU)
2. **양방향 가역 전이** (입자 ↔ surface reservoir 왕복)
3. **Cross-solver heterogeneous coupling** (Eulerian-grid MPM ↔ Lagrangian non-manifold XPBD)
4. **JKR-as-XPBD-constraint** (force가 아닌 phase-transition criterion으로)
5. **Dynamic mass-bearing reservoir** (단순 visual decal 아니라 천의 동역학을 변화시키는 state variable)

상세 비교는 [계획서.md](계획서.md) §2.2 prior art matrix 참고.

## 폴더 구성

```
term_project/
├── README.md            # 이 파일 (개요 + 설치/실행)
├── 계획서.md             # 학기 계획서 v2 (W2-pivot, JKR-as-constraint)
├── 설명서.md             # 환경 셋업 + 12-step 단계별 매뉴얼 v2
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   ├── mpm/             # MLS-MPM sand (Drucker-Prager) + Eulerian grid
│   ├── cloth/           # XPBD cloth + per-triangle σ_front/back + p_σ vertex reservoir
│   ├── coupling/        # JKR closed-form, contact, Operator A (attach)
│   └── utils/           # conservation audit, σ visualization, YAML config
├── data/configs/        # YAML scene + JKR parameter sets
├── scripts/             # 5개 entry point (env check / sand / cloth / contact / attach)
├── tests/               # JKR formula + import + conservation sanity
├── notebooks/           # JKR curve exploration
└── results/             # 실험 결과 (gitignored)
```

## 현재 구현 단계 (MVP)

✅ **Step 1-5** (이 시점):
- Step 1: 환경 점검 (`scripts/check_env.py`)
- Step 2: Sand-only MPM baseline (`scripts/run_sand_only.py`)
- Step 3: Cloth-only XPBD baseline (`scripts/run_cloth_only.py`)
- Step 4: 두 솔버 통합 contact only (`scripts/run_unified_contact.py`)
- **Step 5: Operator A — JKR phase-transition attach** (`scripts/run_attach_demo.py`) ⭐ 핵심 novelty

🚧 **다음 세션 (Step 6 이후)**:
- Step 6: Go/No-Go gate 정량 검증
- Step 7: Effective mass + p_σ-driven swing period shift 측정
- Step 8: **Operator B — Maugis-Dugdale hysteretic detach** (release)
- Step 9: Mass + linear momentum dual conservation theorem proof + audit
- Step 10: 5 demos + 4 ablation baselines (B1 heuristic / B2 world-frame σ / B3 force-based JKR / B4 decal-only)

## 설치

WSL2 Ubuntu 22.04 / native Linux 권장. macOS/Windows는 Taichi GPU 모드만 가능.

```bash
git clone <this-repo-url> realtime-cloth-granular
cd realtime-cloth-granular

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# editable install (모듈 import 경로 단순화)
pip install -e .
```

GPU 검증:

```bash
python3 scripts/check_env.py     # CUDA 인식 + Taichi kernel 실행
```

CUDA가 없으면 자동 CPU fallback (느려도 동작).

## 실행

각 스크립트는 독립 실행 가능. config 파일은 `data/configs/`에 있습니다.

```bash
# Step 2 — sand only
python3 scripts/run_sand_only.py --config data/configs/default_jkr.yaml

# Step 3 — cloth only
python3 scripts/run_cloth_only.py --config data/configs/default_jkr.yaml

# Step 4 — sand + cloth contact (no transfer)
python3 scripts/run_unified_contact.py --config data/configs/demo_a_lying.yaml

# Step 5 — JKR attach demo (핵심)
python3 scripts/run_attach_demo.py \
  --config data/configs/demo_a_lying.yaml \
  --out results/attach_demo_log.npz
```

`run_attach_demo.py`는 다음을 출력합니다:
- 실행 중: `n_active`, `n_attached_total`, mass drift, attach event count per substep
- 종료 시: `results/attach_demo_log.npz` (timeseries + per-vertex σ snapshot)

## 테스트

```bash
python3 -m pytest tests/ -q
```

CPU mode에서 실행됨 — GPU 없이도 통과해야 합니다. 검증 항목:
- `test_imports.py` — 모든 모듈 import + Taichi 초기화 + YAML config 로드
- `test_jkr_formula.py` — JKR closed-form 수치 검증 (numpy 참조 vs Taichi 커널, R²>0.9999)
- `test_conservation.py` — Operator A 발동 시 total mass 보존 (rel err < 1e-5)

## 스케일 조정 (GPU별)

`data/configs/default_jkr.yaml`에서 다음을 수정:

| GPU | `max_particles` | `n_active_particles` | `grid_res` | `cloth_nx × cloth_ny` |
|---|---:|---:|---:|---:|
| RTX 3070 8GB (개발) | 50,000 | 5,000 | 64 | 32 × 32 |
| RTX 4080 16GB (타깃) | 1,000,000 | 100,000 | 128 | 96 × 96 |

## 핵심 결정 추적

설계 결정의 *왜*는 [계획서.md](계획서.md)에 있습니다. 특히:
- §2.2 prior art matrix — 16개 인접 연구와의 명시적 차별
- §2.3 *주장하지 않는 것* (precommit negation list)
- §4.2~4.6 Operator A/B 형식화 + conservation theorem
- §6 Go/No-Go gate 정량 기준
- §11 verified novelty methodology

[설명서.md](설명서.md) Step 0~12은 환경 → baselines → operators → conservation → demos → paper draft까지 순서.

## 다음 세션 인계 사항

집에서 RTX 4080 환경에서 받은 후:

1. `pip install -r requirements.txt && pip install -e .`
2. `python3 scripts/check_env.py` → CUDA 인식 확인
3. `python3 -m pytest tests/ -q` → 모든 테스트 통과 확인
4. 4개 runner 순서대로 실행 → 각 단계 동작 검증
5. 결과 보고 → 다음 세션에 Operator B (Step 8) + ablation (Step 10) 진행

## 라이선스

MIT (학기말 코드 공개 명시).

## 참고

- [Real-time wet sand (Rungjiratananon et al., PG 2008)](https://kanamori.cs.tsukuba.ac.jp/publications/pg08/sand-pg08-fin.pdf) — 가장 가까운 real-time 선례 (cloth 결합 부재)
- [Wet Cloth (Fei et al. 2018)](https://www.cs.columbia.edu/cg/wetcloth/) — 학술 ancestor (fluid 한정, offline)
- [Dynamic Duo (Li et al. 2024)](https://wanghmin.github.io/publication/li-2024-add/) — 가장 가까운 위협 (FEM cloth↔MPM IPC contact, offline)
- [PB-MPM (Lewin/EA SEED 2024)](https://www.ea.com/seed/news/siggraph2024-pbmpm) — real-time MPM platform
- [JKR contact model (Johnson, Kendall, Roberts 1971)](https://royalsocietypublishing.org/doi/10.1098/rspa.1971.0141) — adhesion physics 원전
- [Maugis 1992 (J. Colloid Interface Sci.)](https://www.sciencedirect.com/science/article/abs/pii/0021979792903078) — Maugis-Dugdale regularization (Operator B에서 사용 예정)
