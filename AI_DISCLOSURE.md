# AI Disclosure — AIOps (DA3408) Assignment 1

Kiran Kumar P

## 1. Which tools were used

Claude (Anthropic), used through the Claude Code CLI.

## 2. How they were used

Used only for **code generation** — the scripts and notebooks in this repository:

- **Q2 (`q2_mlflow_mnist/train.py`, `Q2_MLflow_MNIST.ipynb`)** — MLflow tracking boilerplate: `mlflow.set_tracking_uri` / `set_experiment` setup, the `log_param` / `log_metric` / `log_artifact` calls inside the run loop, and the matplotlib code that draws the loss and accuracy curves.
- **Q3 (`q3_dvc_versioning/make_labels.py`)** — the `rglob` + `csv.writer` loop that walks `data/` and writes `labels.csv`.
- **Q3 (DVC setup)** — `dvc init` and `dvc remote add` command syntax for the S3 remote, and the shell one-liner used to capture the v2 → v1 → v2 rollback transcript.
- **Q4 (`prepare_data.py`, `train_and_register.ipynb`)** — the OpenML fetch and `np.savez_compressed` script, and the MLflow model-registration and run-tagging calls.

AI was **not** used for the written work. The technical-debt diagnosis in `q1_technical_debt/Answer.pdf` and the experiment analysis in `q2_mlflow_mnist/Answer.pdf` — the choice of debt categories, the reading of the results, and the argument that learning rate dominates batch size — are my own.

The hyperparameter grid, the two dataset versions, the rollback procedure, and the Partner A / Partner B reproducibility protocol were my design decisions. Every generated snippet was read, edited to fit the assignment, and run before being committed; I can explain and modify all of it.

## 3. Impact

The tool removed setup and syntax friction — MLflow logging calls, DVC command syntax, plotting code — so the time went into the experiments and the analysis instead. It did not change any result: all metrics in this repository come from runs I executed myself, and the conclusions are my own reading of those runs.
