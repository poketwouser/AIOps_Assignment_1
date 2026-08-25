import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--data_path", type=str, default="")  # unused placeholder, mirrors slide example
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    # NOTE: we deliberately do NOT call mlflow.set_experiment() here.
    # `mlflow run` already creates and activates a run inside the experiment
    # you pass via `--experiment-name` on the command line (Step 4 below).
    # Calling set_experiment() here would point at a DIFFERENT experiment than
    # the one `mlflow run` already activated, and mlflow.start_run() would error.

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"project-run-n{args.n_estimators}-d{args.max_depth}"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(model, name="model")

        print(f"accuracy={acc:.4f}  f1_macro={f1:.4f}  run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
