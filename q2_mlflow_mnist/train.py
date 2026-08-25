import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# MLPClassifier(max_iter=1) warns on every .fit() call by design; we drive the
# epoch loop ourselves via warm_start, so the warning carries no information.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_mnist(n_train, seed):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype("float32") / 255.0  # scaling is mandatory: unscaled MNIST makes
    y = y.astype(int)                # every run look equally bad
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=n_train, test_size=2000, stratify=y, random_state=seed
    )
    return X_train, X_val, y_train, y_val


def make_curve_figure(loss_history, val_acc_history):
    fig, ax_loss = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(loss_history) + 1)

    ax_loss.plot(epochs, loss_history, color="tab:red", label="train_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train_loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, val_acc_history, color="tab:blue", label="val_accuracy")
    ax_acc.set_ylabel("val_accuracy", color="tab:blue")
    ax_acc.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("train_loss vs val_accuracy")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)  # SWEPT
    parser.add_argument("--batch_size", type=int, default=64)         # SWEPT
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_curves", action="store_true", default=True)
    parser.add_argument("--no_log_curves", dest="log_curves", action="store_false")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    # We invoke this script with `python train.py` (not `mlflow run`), so nothing
    # has activated an experiment for us -- set it explicitly.
    mlflow.set_experiment("mnist-mlp")

    X_train, X_val, y_train, y_val = load_mnist(args.n_train, args.seed)

    run_name = f"mlp-lr{args.learning_rate}-bs{args.batch_size}"
    with mlflow.start_run(run_name=run_name):
        # ---- params: logged once, before the loop ----
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("n_train", args.n_train)
        mlflow.log_param("solver", "adam")
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("model_type", "MLPClassifier")
        mlflow.log_param("dataset", "MNIST")

        model = MLPClassifier(
            hidden_layer_sizes=(args.hidden_size,),
            learning_rate_init=args.learning_rate,
            batch_size=args.batch_size,
            max_iter=1,        # one epoch per .fit() call
            warm_start=True,   # retains weights AND the Adam optimiser state
            solver="adam",
            random_state=args.seed,
        )

        loss_history, train_acc_history, val_acc_history = [], [], []

        for epoch in range(args.epochs):
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))

            # ---- time-series metrics: step= is what makes these plottable curves ----
            mlflow.log_metric("train_loss", model.loss_, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            loss_history.append(model.loss_)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            print(
                f"epoch {epoch + 1:3d}/{args.epochs}  "
                f"train_loss={model.loss_:.4f}  "
                f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
            )

        # ---- scalar summaries: the columns the comparison table sorts on ----
        val_preds = model.predict(X_val)
        final_val_f1 = f1_score(y_val, val_preds, average="macro")

        mlflow.log_metric("final_train_loss", loss_history[-1])
        mlflow.log_metric("final_train_accuracy", train_acc_history[-1])
        mlflow.log_metric("final_val_accuracy", val_acc_history[-1])
        mlflow.log_metric("best_val_accuracy", max(val_acc_history))
        mlflow.log_metric("final_val_f1_macro", final_val_f1)
        mlflow.log_metric("overfit_gap", train_acc_history[-1] - val_acc_history[-1])

        if args.log_curves:
            fig = make_curve_figure(loss_history, val_acc_history)
            mlflow.log_figure(fig, "loss_vs_valacc.png")
            plt.close(fig)

        # warm_start keeps the Adam optimiser state on the estimator, and MLflow's
        # skops serialiser refuses to save that type unless it is declared trusted.
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
            ],
        )

        print(
            f"final_val_accuracy={val_acc_history[-1]:.4f}  "
            f"best_val_accuracy={max(val_acc_history):.4f}  "
            f"final_val_f1_macro={final_val_f1:.4f}  "
            f"overfit_gap={train_acc_history[-1] - val_acc_history[-1]:.4f}  "
            f"run_id={mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()
