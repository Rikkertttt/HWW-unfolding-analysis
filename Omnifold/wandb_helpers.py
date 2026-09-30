import wandb
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Mapping


def init_omnifold_run(
    iterations: int,
    theta0: np.ndarray,
    theta_unknown_S: np.ndarray,
    extra_config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> None:
    config: Dict[str, Any] = {
        "iterations": iterations,
        "features":   theta0.shape[-1],
        "n_sim":      len(theta0),
        "n_data":     len(theta_unknown_S),
    }
    if extra_config:
        config.update(extra_config)

    wandb.init(
        entity="rikvanrhee-nikhef-master",
        project="hww-omnifold",
        name=run_name,
        config=config,
    )

    wandb.define_metric("epoch")
    for i in range(1, iterations + 1):
        for s in (1, 2):
            wandb.define_metric(f"iter{i}/step{s}/*", step_metric="epoch")

    wandb.define_metric("iteration")
    wandb.define_metric("weights/*", step_metric="iteration")


class WandbOmniFoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, iteration: int, step: int, epochs: int = 20) -> None:
        super().__init__()
        self.iteration = iteration
        self.step = step
        self.epochs = epochs

    def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        logs = logs or {}
        wandb.log({
            f"iter{self.iteration+1}/step{self.step}/loss":         logs.get("loss"),
            f"iter{self.iteration+1}/step{self.step}/val_loss":     logs.get("val_loss"),
            f"iter{self.iteration+1}/step{self.step}/accuracy":     logs.get("accuracy"),
            f"iter{self.iteration+1}/step{self.step}/val_accuracy": logs.get("val_accuracy"),
            "epoch": epoch,
        })


def log_weights(
    iteration: int,
    weights_pull: np.ndarray,
    weights_push: np.ndarray,
) -> None:
    wandb.log({
        "weights/pull_mean": weights_pull.mean(),
        "weights/pull_std":  weights_pull.std(),
        "weights/push_mean": weights_push.mean(),
        "weights/push_std":  weights_push.std(),
        "iteration":         iteration + 1,
    })