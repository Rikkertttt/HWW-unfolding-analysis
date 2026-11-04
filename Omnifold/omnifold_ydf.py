# Yggdrasil Decision Forest (YDF) implementation of Omnifold.

import os
import numpy as np
import ydf
from typing import Optional
import matplotlib.pyplot as plt

def reweight_ydf(events_dict: dict[str, np.ndarray], model: ydf.GradientBoostedTreesModel) -> np.ndarray:
    probs = model.predict(events_dict)   # shape (N,), already P(label="1")
    weights = probs / (1. - probs)
    return np.nan_to_num(weights)

def to_ydf_ds(X: np.ndarray, feature_names: list[str], y: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    ds = {name: X[:, i] for i, name in enumerate(feature_names)}
    ds["label"] = y.astype(str)   # must be string for CATEGORICAL
    ds["weight"] = w
    return ds

def _save_loss_plot(model: ydf.GradientBoostedTreesModel, report_dir: str, i: int, step: int) -> None:
    logs = model.training_logs()
    iterations  = [log.iteration for log in logs]
    valid_loss  = [log.evaluation.loss for log in logs]
    train_loss  = [log.training_evaluation.loss if log.training_evaluation is not None else float("nan") for log in logs]
    valid_acc   = [log.evaluation.accuracy for log in logs]
    train_acc   = [log.training_evaluation.accuracy if log.training_evaluation is not None else float("nan") for log in logs]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(iterations, train_loss, label="train")
    ax_loss.plot(iterations, valid_loss, label="valid")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"Loss — Iter {i + 1}, Step {step}")
    ax_loss.legend()

    ax_acc.plot(iterations, train_acc, label="train")
    ax_acc.plot(iterations, valid_acc, label="valid")
    ax_acc.set_xlabel("iteration")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"Accuracy — Iter {i + 1}, Step {step}")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(f"{report_dir}/loss_iter{i}_step{step}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def omnifold_ydf(theta0: np.ndarray,
                 theta_unknown_S: np.ndarray,
                 iterations: int,
                 feature_names: Optional[list[str]] = None,
                 verbose: int = 0,
                 save_report_path: Optional[str] = None,
                 num_trees: int = 300,) -> np.ndarray:

    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)

    theta0_G = theta0[:, 0]
    theta0_S = theta0[:, 1]

    labels0 = np.zeros(len(theta0))
    labels_unknown = np.ones(len(theta_unknown_S))
    labels_unknown_step2 = np.ones(len(theta0_G))

    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))

    if feature_names is None:
        feature_names = ["f{}".format(i) for i in range(theta0.shape[2])]

    report_dir = None
    if save_report_path is not None:
        report_dir = os.path.join(save_report_path, "ydf_reports")
        os.makedirs(report_dir, exist_ok=True)

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))

    for i in range(iterations):

        if verbose > 0:
            print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if verbose > 0:
            print("STEP 1\n")

        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        ds_1 = to_ydf_ds(X=xvals_1, feature_names=feature_names, y=yvals_1, w=weights_1)
        learner = ydf.GradientBoostedTreesLearner(label="label",
                                                  weights="weight",
                                                  discretize_numerical_columns=True,
                                                  num_trees=num_trees)
        model = learner.train(ds_1)

        weights_pull = weights_push * reweight_ydf(
            events_dict={name: theta0_S[:, j] for j, name in enumerate(feature_names)},
            model=model)
        weights[i, :1, :] = weights_pull

        if report_dir is not None:
            with open(f"{report_dir}/model_iter{i}_step1.html", "w") as f:
                f.write(model.describe())  # type: ignore
            _save_loss_plot(model, report_dir, i, step=1)

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if verbose > 0:
            print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        ds_2 = to_ydf_ds(X=xvals_2, feature_names=feature_names, y=yvals_2, w=weights_2)
        learner = ydf.GradientBoostedTreesLearner(label="label",
                                                  weights="weight",
                                                  discretize_numerical_columns=True,
                                                  num_trees=num_trees)
        model = learner.train(ds_2)

        weights_push = reweight_ydf(
            events_dict={name: theta0_G[:, j] for j, name in enumerate(feature_names)},
            model=model)
        weights[i, 1:2, :] = weights_push

        if report_dir is not None:
            with open(f"{report_dir}/model_iter{i}_step2.html", "w") as f:
                f.write(model.describe())  # type: ignore
            _save_loss_plot(model, report_dir, i, step=2)

    return weights