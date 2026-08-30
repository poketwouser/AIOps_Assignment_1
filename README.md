# AI Operations (DA3408) — Assignment 1

**Kiran Kumar P**

Repo: <https://github.com/poketwouser/AIOps_Assignment_1>
Question 4 lives in a separate partner repo: <https://github.com/Kevii137/AIops-A1Q4>

Summary write-up for all four questions: **[`Report.pdf`](Report.pdf)** (1 page).
AI use declaration: **[`AI_DISCLOSURE.md`](AI_DISCLOSURE.md)** (course code-of-conduct §4).
Per-question write-ups are the `Answer.pdf` files listed below.

---

## Where to look — deliverables map

### Q1 — Technical debt diagnosis
| Deliverable | File |
|---|---|
| Three hidden-technical-debt categories identified & justified | [`q1_technical_debt/Answer.pdf`](q1_technical_debt/Answer.pdf) §1.1 |
| Mitigation for category (c) with a named MLOps tool | [`q1_technical_debt/Answer.pdf`](q1_technical_debt/Answer.pdf) §1.2 |

Categories argued: **Entanglement (CACE)**, **Undeclared Consumers**, **Configuration & Glue-Code Debt**.
Mitigation: MLflow Projects — an `MLproject` file with declared entry points + pinned `env.yaml`, run via `mlflow run`.

### Q2 — MLflow experiment comparison (MLP on MNIST)
| Deliverable | File |
|---|---|
| Training code (6 runs, MLflow-tracked) | [`q2_mlflow_mnist/train.py`](q2_mlflow_mnist/train.py) |
| Driver notebook + inline outputs | [`q2_mlflow_mnist/Q2_MLflow_MNIST.ipynb`](q2_mlflow_mnist/Q2_MLflow_MNIST.ipynb) |
| **Written analysis** (results table, best run, overfitting evidence, larger-effect argument) | [`q2_mlflow_mnist/Answer.pdf`](q2_mlflow_mnist/Answer.pdf) |
| Exported run table (all 6 runs, all metrics) | [`q2_mlflow_mnist/results/runs.csv`](q2_mlflow_mnist/results/runs.csv) |
| MLflow UI — run comparison screenshots | `q2_mlflow_mnist/results/Results - 1.png`, `Results - 2.png` |
| MLflow UI — loss / accuracy curves | `q2_mlflow_mnist/results/Visualizations - 1.png`, `Visualizations - 2.png` |
| MLflow parallel-coordinates plot (lr vs bs effect) | `q2_mlflow_mnist/results/Parallel Coordinate Plot.png` |

Grid: `learning_rate ∈ {0.001, 0.01, 0.1}` × `batch_size ∈ {32, 256}`.
Fixed: 128 hidden units, 30 epochs, Adam, seed 42, 10,000 train / 2,000 val.
**Best run:** `lr=0.001, bs=32` → `final_val_accuracy = 0.9525` (best 0.9545), `final_train_loss = 0.0053`.

> `mlruns/` and `mlflow.db` are gitignored (regenerable local tracking state). The evidence is `results/runs.csv` + the screenshots.

### Q3 — DVC data versioning & rollback
| Deliverable | File / evidence |
|---|---|
| Dataset build script (deterministic) | [`q3_dvc_versioning/make_labels.py`](q3_dvc_versioning/make_labels.py) |
| DVC pointer files (data is **not** in Git) | [`q3_dvc_versioning/data.dvc`](q3_dvc_versioning/data.dvc), [`q3_dvc_versioning/labels.csv.dvc`](q3_dvc_versioning/labels.csv.dvc) |
| Remote storage config (AWS S3) | [`.dvc/config`](.dvc/config) → `s3://aiops-kiran-a1q3/q3-dvc` |
| Proof the remote holds the data | `q3_dvc_versioning/rollback_proofs/S3 Remote.png` |
| **Rollback transcript (v2 → v1 → v2)** | [`q3_dvc_versioning/rollback_proofs/rollback_proof.txt`](q3_dvc_versioning/rollback_proofs/rollback_proof.txt) |
| Rollback screenshots | `q3_dvc_versioning/rollback_proofs/Proof - 1.png`, `Proof - 2.png` |

Versions are Git tags:

| Tag | Commit | Images | `labels.csv` |
|---|---|---|---|
| `v1` | `8bb78c1` | 1800 | 1801 lines (1800 rows + header) |
| `v2` | `ffbd981` | 2800 | 2801 lines (2800 rows + header) |

Reproduce the rollback:
```bash
git checkout v1 && dvc checkout   # workspace drops to 1800 images / 1801 lines
git checkout main && dvc checkout # back to 2800 images / 2801 lines
```
The transcript shows the key point: `git checkout v1` alone moves **pointers only** (`labels.csv` still 2801 lines); the workspace only changes after `dvc checkout`.

### Q4 — End-to-end reproducibility capstone (partner exercise)
Separate repository: **<https://github.com/Kevii137/AIops-A1Q4>** — see its `README.md` for the full A/B comparison table.

| Deliverable | Where in that repo |
|---|---|
| Training + registration notebook | `train_and_register.ipynb` |
| Dataset fetch script | `prepare_data.py` |
| Data versioned in DVC (S3 remote) | `data.dvc`, `.dvc/config` → `s3://aiops-kevin-2026/repro-handoff` |
| Pinned environment | `requirements.txt`, `run_environment.json` |
| Partner A reference result & Partner B reproduction verdict | `README.md` |
| MLflow proof screenshots | `proofs/Reproduction Tags.png`, `Reproduction Note.png`, `Metrics Comparision.png`, `Parallel Plot.png` |

**Verdict: `MATCHED`** — Partner A `0.9771428571428571` vs Partner B `0.9771428571428571`, delta `+0.0000` against a `±0.005` tolerance. Model registered as `mnist-mlp`, stage `Staging`, pinned to commit `daa1875`.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Conda alternative: `conda env create -f environment.yml` (`environment.yml`, Python 3.12).

Pinned: `mlflow==3.15.1`, `scikit-learn==1.9.0`, `numpy==2.5.2`, `pandas==2.3.3`, `matplotlib==3.11.1`, `dvc==3.67.1`, `dvc-s3==3.3.0`.

## Running things

```bash
# Q2 — start tracking server, then run the grid
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
python q2_mlflow_mnist/train.py          # UI at http://localhost:5000

# Q3 — pull the versioned data from S3 (needs AWS creds)
dvc pull
```

## Layout

```
Assignment_1/
├── README.md                 <- this file
├── Report.pdf                <- 1-page summary, all 4 questions
├── requirements.txt / environment.yml
├── .dvc/config               <- S3 remote for Q3
├── q1_technical_debt/Answer.pdf
├── q2_mlflow_mnist/          <- train.py, notebook, Answer.pdf, results/
└── q3_dvc_versioning/        <- make_labels.py, *.dvc, rollback_proofs/
```
`data/`, `labels.csv`, `mlruns/`, `mlflow.db` and `.venv/` are gitignored — data is restored with `dvc pull`, MLflow state is regenerable.
